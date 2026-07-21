#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import pyarrow as pa

from tools import analyze_ingress_rrd as analyzer


class _Reader:
    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def to_arrow_table(self) -> pa.Table:
        return self._table


class _View:
    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def reader(self, *, index: str | None) -> _Reader:
        del index
        return _Reader(self._table)


class _Dataset:
    def __init__(self, tables: dict[str, pa.Table]) -> None:
        self._tables = tables

    def filter_contents(self, path: str) -> _View:
        return _View(self._tables.get(path, pa.table({})))


class _IndexAwareView:
    def __init__(self, tables: dict[str, pa.Table]) -> None:
        self._tables = tables

    def reader(self, *, index: str | None) -> _Reader:
        return _Reader(self._tables.get(str(index), pa.table({})))


class _IndexAwareDataset:
    def __init__(self, tables: dict[str, dict[str, pa.Table]]) -> None:
        self._tables = tables

    def filter_contents(self, path: str) -> _IndexAwareView:
        return _IndexAwareView(self._tables.get(path, {}))


def _descriptor(path: str, name: str, component_type: str) -> Any:
    return SimpleNamespace(
        entity_path=path,
        name=name,
        component_type=component_type,
        is_property=False,
    )


def _scalar_table(
    column: str, values: list[list[float]], **timelines: list[int]
) -> pa.Table:
    return pa.table(
        {
            **timelines,
            column: pa.array(values, type=pa.list_(pa.float64())),
        }
    )


def _point_table(
    column: str, frame_sizes: list[int], **timelines: list[int]
) -> pa.Table:
    frames = [
        [[float(index), 0.0, 0.0] for index in range(size)]
        for size in frame_sizes
    ]
    return pa.table(
        {
            **timelines,
            column: pa.array(
                frames,
                type=pa.list_(pa.list_(pa.float32(), 3)),
            ),
        }
    )


class LocalRtReportTest(unittest.TestCase):
    def test_ingress_only_dataset_remains_optional(self) -> None:
        report = analyzer._local_rt_report(
            _Dataset({}), [], {"measurement_id", "sensor_time", "state_id", "log_time"}
        )

        self.assertFalse(report["present"])
        self.assertFalse(report["initialization"]["present"])
        self.assertEqual(report["initialization"]["records"], 0)
        self.assertFalse(report["initialization"]["accepted_seed_present"])
        self.assertEqual(report["bootstrap"]["records"], 0)
        self.assertEqual(report["preintegration"]["records"], 0)
        self.assertIsNone(report["preintegration"]["backend"])

    def test_summarizes_initialization_bootstrap_and_preintegration(self) -> None:
        status_column = "initialization_status"
        counts_column = "initialization_counts"
        seed_column = "seed_velocity"
        bootstrap_column = "bootstrap_quality"
        preintegration_column = "preintegration_quality"
        backend_column = "preintegration_backend"
        descriptors = [
            _descriptor(
                analyzer.INITIALIZATION_STATUS_PATH,
                status_column,
                analyzer.TEXT_COMPONENT,
            ),
            _descriptor(
                analyzer.INITIALIZATION_COUNTS_PATH,
                counts_column,
                analyzer.SCALAR_COMPONENT,
            ),
            _descriptor(
                analyzer.ACCEPTED_SEED_PATH,
                seed_column,
                analyzer.SCALAR_COMPONENT,
            ),
            _descriptor(
                analyzer.BOOTSTRAP_QUALITY_PATH,
                bootstrap_column,
                analyzer.SCALAR_COMPONENT,
            ),
            _descriptor(
                analyzer.PREINTEGRATION_QUALITY_PATH,
                preintegration_column,
                analyzer.SCALAR_COMPONENT,
            ),
            _descriptor(
                analyzer.PREINTEGRATION_BACKEND_PATH,
                backend_column,
                analyzer.TEXT_COMPONENT,
            ),
        ]
        tables = {
            analyzer.INITIALIZATION_STATUS_PATH: pa.table(
                {
                    "sensor_time": [100, 200],
                    status_column: [
                        "mode=STATIC status=COLLECTING reason=waiting for support",
                        "mode=STATIC status=ACCEPTED reason=static gates passed",
                    ],
                }
            ),
            analyzer.INITIALIZATION_COUNTS_PATH: _scalar_table(
                counts_column,
                [[100.0, 0.0, 0.0, 0.0], [420.0, 0.0, 0.0, 0.0]],
                sensor_time=[100, 200],
            ),
            analyzer.ACCEPTED_SEED_PATH: _scalar_table(
                seed_column,
                [[0.0, 0.0, 0.0]],
                sensor_time=[200],
                state_id=[1],
            ),
            analyzer.BOOTSTRAP_QUALITY_PATH: _scalar_table(
                bootstrap_column,
                [
                    [1_000.0, 800.0, 0.10, 20.0, 1.0],
                    [900.0, 300.0, 0.40, 50.0, 0.0],
                ],
                sensor_time=[300, 400],
                measurement_id=[7, 8],
            ),
            analyzer.PREINTEGRATION_QUALITY_PATH: _scalar_table(
                preintegration_column,
                [
                    [1.0, 2.0, 100_000_000.0, 21.0, 20.0, 5_000_000.0],
                    [2.0, 3.0, 200_000_000.0, 41.0, 40.0, 8_000_000.0],
                ],
                sensor_time=[1_100_000_000, 1_300_000_000],
                state_id=[2, 3],
            ),
            analyzer.PREINTEGRATION_BACKEND_PATH: pa.table(
                {
                    "sensor_time": [1_100_000_000, 1_300_000_000],
                    "state_id": [2, 3],
                    backend_column: [
                        "gtsam-resolved-tangent",
                        "gtsam-resolved-tangent",
                    ],
                }
            ),
        }

        report = analyzer._local_rt_report(
            _Dataset(tables),
            descriptors,
            {"measurement_id", "sensor_time", "state_id", "log_time"},
        )

        self.assertTrue(report["present"])
        initialization = report["initialization"]
        self.assertEqual(initialization["records"], 2)
        self.assertEqual(initialization["final"]["mode"], "STATIC")
        self.assertEqual(initialization["final"]["status"], "ACCEPTED")
        self.assertEqual(initialization["final"]["reason"], "static gates passed")
        self.assertTrue(initialization["accepted_seed_present"])
        self.assertEqual(initialization["final_counts"]["imu_sample_count"], 420)

        bootstrap = report["bootstrap"]
        self.assertEqual(bootstrap["records"], 2)
        self.assertEqual(bootstrap["accepted_records"], 1)
        self.assertEqual(bootstrap["rejected_records"], 1)

        preintegration = report["preintegration"]
        self.assertEqual(preintegration["records"], 2)
        self.assertEqual(preintegration["backend"], "gtsam-resolved-tangent")
        self.assertEqual(preintegration["backends"], ["gtsam-resolved-tangent"])
        self.assertEqual(preintegration["support"]["first_begin_ns"], 1_000_000_000)
        self.assertEqual(preintegration["support"]["last_end_ns"], 1_300_000_000)
        self.assertAlmostEqual(preintegration["support"]["total_duration_s"], 0.3)
        self.assertEqual(preintegration["source_sample_count"]["max"], 41.0)
        self.assertEqual(preintegration["integration_segment_count"]["mean"], 30.0)
        self.assertEqual(preintegration["maximum_source_gap_ns"]["max"], 8_000_000.0)
        self.assertEqual(preintegration["state_id_unique"], 2)

    def test_status_parser_preserves_unstructured_text(self) -> None:
        parsed = analyzer._parse_initialization_status("unexpected status text")
        self.assertEqual(parsed["raw"], "unexpected status text")
        self.assertIsNone(parsed["mode"])
        self.assertIsNone(parsed["status"])
        self.assertIsNone(parsed["reason"])


class TimingReportTest(unittest.TestCase):
    def test_discovers_stages_and_reports_tail_latency(self) -> None:
        deskew_path = analyzer.TIMING_PREFIX + "deskew"
        solve_path = analyzer.TIMING_PREFIX + "window_solve"
        deskew_column = "deskew_duration"
        solve_column = "solve_duration"
        descriptors = [
            _descriptor(deskew_path, deskew_column, analyzer.SCALAR_COMPONENT),
            _descriptor(solve_path, solve_column, analyzer.SCALAR_COMPONENT),
        ]
        tables = {
            deskew_path: _scalar_table(
                deskew_column,
                [[10.0], [20.0], [30.0], [40.0], [50.0]],
                sensor_time=[100, 200, 300, 400, 500],
                state_id=[1, 2, 3, 4, 5],
            ),
            solve_path: _scalar_table(
                solve_column,
                [[100.0], [-1.0]],
                sensor_time=[200, 300],
                state_id=[2, 3],
            ),
        }

        report = analyzer._timing_report(
            _Dataset(tables), descriptors, {"sensor_time", "state_id", "log_time"}
        )

        self.assertTrue(report["present"])
        self.assertEqual(report["stage_count"], 2)
        deskew = report["stages"]["deskew"]
        self.assertEqual(deskew["duration_ns"]["count"], 5)
        self.assertEqual(deskew["duration_ns"]["median"], 30.0)
        self.assertEqual(deskew["duration_ns"]["p95"], 48.0)
        self.assertAlmostEqual(deskew["duration_ns"]["p99"], 49.6)
        self.assertEqual(deskew["duration_ns"]["max"], 50.0)
        self.assertEqual(deskew["timeline"]["state_id_unique"], 5)
        solve = report["stages"]["window_solve"]
        self.assertEqual(solve["valid_duration_rows"], 1)
        self.assertEqual(solve["invalid_rows"], 1)

    def test_absent_timing_is_optional(self) -> None:
        report = analyzer._timing_report(_Dataset({}), [], {"sensor_time"})
        self.assertFalse(report["present"])
        self.assertEqual(report["stage_count"], 0)
        self.assertEqual(report["stages"], {})

    def test_sensor_time_index_preserves_repeated_rejected_state_attempts(self) -> None:
        path = analyzer.TIMING_PREFIX + "window_solve"
        column = "window_solve_duration"
        # This mirrors Rerun dataframe behavior: selecting state_id keeps the
        # latest row for state 2, while selecting sensor_time retains all three
        # rejected attempts that reused state 2 plus the accepted state 3 row.
        sensor_time_table = _scalar_table(
            column,
            [[10.0], [20.0], [30.0], [40.0]],
            sensor_time=[100, 200, 300, 400],
            state_id=[2, 2, 2, 3],
        )
        state_id_table = _scalar_table(
            column,
            [[30.0], [40.0]],
            sensor_time=[300, 400],
            state_id=[2, 3],
        )
        dataset = _IndexAwareDataset(
            {path: {"sensor_time": sensor_time_table, "state_id": state_id_table}}
        )
        descriptors = [_descriptor(path, column, analyzer.SCALAR_COMPONENT)]

        report = analyzer._timing_report(
            dataset, descriptors, {"sensor_time", "state_id", "log_time"}
        )

        stage = report["stages"]["window_solve"]
        self.assertEqual(stage["query_index"], "sensor_time")
        self.assertEqual(stage["records"], 4)
        self.assertEqual(stage["duration_ns"]["count"], 4)
        self.assertEqual(stage["duration_ns"]["mean"], 25.0)
        self.assertEqual(stage["timeline"]["state_id_rows"], 4)
        self.assertEqual(stage["timeline"]["state_id_unique"], 2)


class RegistrationMapReportTest(unittest.TestCase):
    def test_reports_native_cadence_timelines_and_point_population(self) -> None:
        column = "registration_map_positions"
        sensor_time_table = _point_table(
            column,
            [2, 5, 3],
            sensor_time=[1_000_000_000, 1_200_000_000, 1_400_000_000],
            state_id=[4, 4, 5],
            estimator_revision=[7, 8, 9],
        )
        # Selecting the repeated state ID would collapse the first native
        # snapshot. The analyzer must instead query on sensor_time.
        state_id_table = _point_table(
            column,
            [5, 3],
            sensor_time=[1_200_000_000, 1_400_000_000],
            state_id=[4, 5],
            estimator_revision=[8, 9],
        )
        dataset = _IndexAwareDataset(
            {
                analyzer.REGISTRATION_MAP_PATH: {
                    "sensor_time": sensor_time_table,
                    "state_id": state_id_table,
                }
            }
        )
        descriptors = [
            _descriptor(
                analyzer.REGISTRATION_MAP_PATH,
                column,
                analyzer.POSITION_COMPONENT,
            )
        ]

        report = analyzer._registration_map_report(
            dataset,
            descriptors,
            {"sensor_time", "state_id", "estimator_revision", "log_time"},
        )

        self.assertTrue(report["present"])
        self.assertEqual(report["query_index"], "sensor_time")
        self.assertEqual(report["frames"], 3)
        self.assertEqual(report["total_points"], 10)
        self.assertEqual(report["timeline"]["sensor_time_first_ns"], 1_000_000_000)
        self.assertEqual(report["timeline"]["sensor_time_last_ns"], 1_400_000_000)
        self.assertAlmostEqual(report["achieved_rate_hz"], 5.0)
        self.assertEqual(report["points_per_frame"]["min"], 2.0)
        self.assertAlmostEqual(report["points_per_frame"]["mean"], 10.0 / 3.0)
        self.assertEqual(report["points_per_frame"]["max"], 5.0)
        self.assertEqual(report["timeline"]["state_id_rows"], 3)
        self.assertEqual(report["timeline"]["estimator_revision_rows"], 3)

    def test_absent_registration_map_is_optional(self) -> None:
        report = analyzer._registration_map_report(
            _Dataset({}), [], {"sensor_time", "state_id", "log_time"}
        )

        self.assertFalse(report["present"])
        self.assertEqual(report["frames"], 0)
        self.assertIsNone(report["points_per_frame"])
        self.assertIsNone(report["achieved_rate_hz"])


class EstimatorQualityReportTest(unittest.TestCase):
    def test_exposes_append_only_scan_to_map_counters(self) -> None:
        column = "estimator_quality"
        sensor_time_table = _scalar_table(
            column,
            [
                [float(value) for value in range(len(analyzer.ESTIMATOR_QUALITY_FIELDS))],
                [
                    float(2 * value)
                    for value in range(len(analyzer.ESTIMATOR_QUALITY_FIELDS))
                ],
            ],
            sensor_time=[100, 200],
            state_id=[3, 3],
            estimator_revision=[10, 11],
        )
        state_id_table = _scalar_table(
            column,
            [[float(2 * value) for value in range(len(analyzer.ESTIMATOR_QUALITY_FIELDS))]],
            sensor_time=[200],
            state_id=[3],
            estimator_revision=[11],
        )
        dataset = _IndexAwareDataset(
            {
                analyzer.ESTIMATOR_QUALITY_PATH: {
                    "sensor_time": sensor_time_table,
                    "state_id": state_id_table,
                }
            }
        )
        descriptors = [
            _descriptor(
                analyzer.ESTIMATOR_QUALITY_PATH,
                column,
                analyzer.SCALAR_COMPONENT,
            )
        ]

        report = analyzer._estimator_quality_report(
            dataset,
            descriptors,
            {"sensor_time", "state_id", "estimator_revision", "log_time"},
        )

        self.assertEqual(report["query_index"], "sensor_time")
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["width_min"], len(analyzer.ESTIMATOR_QUALITY_FIELDS))
        self.assertEqual(report["width_max"], len(analyzer.ESTIMATOR_QUALITY_FIELDS))
        self.assertEqual(report["timeline"]["state_id_rows"], 2)
        self.assertEqual(report["timeline"]["estimator_revision_rows"], 2)
        self.assertEqual(
            report["field_stats"]["prepared_target_points"]["mean"], 22.5
        )
        self.assertEqual(
            report["field_stats"]["live_query_voxel_probes"]["max"], 42.0
        )
        self.assertEqual(
            report["field_stats"]["rejected_stale_rows"]["max"], 48.0
        )


class TrajectoryReportTest(unittest.TestCase):
    @staticmethod
    def _trajectory_dataset(
        timestamps_ns: list[int], positions: list[list[float]]
    ) -> tuple[_Dataset, list[Any]]:
        path = analyzer.TRAJECTORY_PATHS["causal"]
        column = "causal_pose"
        values = [position + [0.0, 0.0, 0.0, 1.0] for position in positions]
        table = _scalar_table(
            column,
            values,
            sensor_time=timestamps_ns,
            state_id=list(range(1, len(values) + 1)),
            estimator_revision=list(range(1, len(values) + 1)),
        )
        return _Dataset({path: table}), [
            _descriptor(path, column, analyzer.SCALAR_COMPONENT)
        ]

    def test_extracts_double_pose_track_and_reports_exact_gap(self) -> None:
        dataset, descriptors = self._trajectory_dataset(
            [1_000_000_000, 1_100_000_000, 1_800_000_000],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        )

        report, trajectories = analyzer._trajectory_report(
            dataset, descriptors, {"sensor_time", "state_id", "log_time"}
        )

        self.assertTrue(report["present"])
        causal = report["tracks"]["causal"]
        self.assertTrue(causal["valid"])
        self.assertEqual(causal["rows"], 3)
        self.assertEqual(causal["width_min"], 7)
        self.assertEqual(causal["timeline"]["estimator_revision_rows"], 3)
        self.assertEqual(causal["timeline"]["gap_count"], 1)
        gap = causal["timeline"]["gap_candidates"][0]
        self.assertEqual(gap["before_state_id"], 2)
        self.assertEqual(gap["after_state_id"], 3)
        self.assertEqual(gap["gap_ns"], 700_000_000)
        self.assertEqual(trajectories["causal"].count, 3)
        self.assertFalse(report["tracks"]["final"]["present"])

    def test_exact_state_csv_drives_existing_nearest_se3_evaluator(self) -> None:
        timestamps_ns = [
            1_625_132_069_100_000_000,
            1_625_132_069_300_000_000,
            1_625_132_069_500_000_000,
            1_625_132_069_700_000_000,
            1_625_132_069_900_000_000,
        ]
        reference_positions = [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
            [4.0, 4.0, 4.0],
            [8.0, 4.0, 4.0],
        ]
        estimate_positions = [
            [point[0] + 10.0, point[1] - 2.0, point[2] + 3.0]
            for point in reference_positions
        ]
        dataset, descriptors = self._trajectory_dataset(timestamps_ns, estimate_positions)
        _, trajectories = analyzer._trajectory_report(
            dataset, descriptors, {"sensor_time", "state_id", "log_time"}
        )

        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "gt-state.csv"
            lines = ["#sec,nsec,x,y,z,qx,qy,qz,qw"]
            for timestamp_ns, position in zip(timestamps_ns, reference_positions):
                seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
                lines.append(
                    f"{seconds},{nanoseconds},{position[0]},{position[1]},"
                    f"{position[2]},0,0,0,1"
                )
            state_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            evaluation = analyzer._evaluate_trajectory_tracks(
                state_path, {"causal": trajectories["causal"]}
            )

        self.assertEqual(evaluation["ground_truth"]["format"], "sec_nsec_xyz_quaternion_xyzw")
        self.assertEqual(evaluation["ground_truth"]["pose_semantics"], "T_world_base")
        self.assertEqual(evaluation["ground_truth"]["first_timestamp_ns"], timestamps_ns[0])
        self.assertEqual(evaluation["evaluated_track_count"], 1)
        metrics = evaluation["tracks"]["causal"]["report"]
        self.assertEqual(metrics["protocol"]["association"], "nearest")
        self.assertEqual(metrics["protocol"]["max_dt_s"], 0.01)
        self.assertAlmostEqual(metrics["ate_translation_m"]["rmse"], 0.0, places=12)
        self.assertTrue(metrics["quality"]["passed"])


if __name__ == "__main__":
    unittest.main()
