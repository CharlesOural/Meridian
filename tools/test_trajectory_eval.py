#!/usr/bin/env python3
"""Synthetic regression tests for the trajectory evaluation protocol."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.trajectory_eval import (
    EvaluationConfig,
    EvaluationError,
    FIXED_COLD_START_PROTOCOL,
    FULL_REFERENCE_PROTOCOL,
    Trajectory,
    _config_from_arguments,
    associate_trajectories,
    build_argument_parser,
    evaluate_trajectories,
    load_scenario_evaluation_config,
    load_tum,
)


def yaw_quaternion(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float64)
    result = np.zeros((yaw.shape[0], 4), dtype=np.float64)
    result[:, 2] = np.sin(0.5 * yaw)
    result[:, 3] = np.cos(0.5 * yaw)
    return result


def trajectory(
    timestamps: np.ndarray,
    positions: np.ndarray,
    yaw: np.ndarray | None = None,
) -> Trajectory:
    stamps = np.asarray(timestamps, dtype=np.float64)
    if yaw is None:
        yaw = np.zeros(stamps.shape[0])
    return Trajectory(stamps, np.asarray(positions, dtype=np.float64), yaw_quaternion(yaw))


def base_frame_config(**overrides: object) -> EvaluationConfig:
    """Build a test configuration with an explicit, valid body-frame pair."""

    return EvaluationConfig(
        reference_body_frame="base_link",
        estimate_body_frame="base_link",
        **overrides,
    )


class TumParsingTest(unittest.TestCase):
    def write(self, text: str) -> Path:
        temporary = tempfile.NamedTemporaryFile(mode="w", suffix=".tum", delete=False)
        temporary.write(text)
        temporary.close()
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        return Path(temporary.name)

    def test_parses_comments_commas_and_normalizes_small_quaternion_error(self) -> None:
        path = self.write(
            "# stamp xyz quaternion\n"
            "0, 1, 2, 3, 0, 0, 0, 1.001 # inline\n"
            "1 2 3 4 0 0 0 1\n"
        )
        parsed = load_tum(path)
        self.assertEqual(parsed.count, 2)
        np.testing.assert_allclose(np.linalg.norm(parsed.quaternions_xyzw, axis=1), 1.0)

    def test_rejects_nan_malformed_duplicate_and_timestamp_reset(self) -> None:
        cases = (
            "0 0 0 0 0 0 0 1\n1 nan 0 0 0 0 0 1\n",
            "0 0 0 0 0 0 0\n",
            "0 0 0 0 0 0 0 1\n0 0 0 0 0 0 0 1\n",
            "1 0 0 0 0 0 0 1\n0 0 0 0 0 0 0 1\n",
        )
        for contents in cases:
            with self.subTest(contents=contents), self.assertRaises(EvaluationError):
                load_tum(self.write(contents))


class AssociationTest(unittest.TestCase):
    def test_interpolates_estimate_at_every_ground_truth_stamp(self) -> None:
        reference_times = np.arange(0.0, 4.01, 0.5)
        estimate_times = np.arange(-0.1, 4.11, 0.2)
        reference = trajectory(
            reference_times,
            np.column_stack((reference_times, 2.0 * reference_times, 0.0 * reference_times)),
            0.1 * reference_times,
        )
        estimate = trajectory(
            estimate_times,
            np.column_stack((estimate_times, 2.0 * estimate_times, 0.0 * estimate_times)),
            0.1 * estimate_times,
        )
        associated = associate_trajectories(
            reference,
            estimate,
            base_frame_config(association="interpolate", max_interpolation_gap=0.21),
        )
        np.testing.assert_array_equal(associated.reference.timestamps, reference_times)
        np.testing.assert_allclose(associated.estimate.positions, reference.positions, atol=1.0e-12)

    def test_nearest_is_one_to_one_and_respects_ten_millisecond_gate(self) -> None:
        reference = trajectory(
            np.array([0.0, 0.01, 0.02]),
            np.zeros((3, 3)),
        )
        estimate = trajectory(
            np.array([0.006, 0.020]),
            np.zeros((2, 3)),
        )
        associated = associate_trajectories(
            reference, estimate, base_frame_config(association="nearest", max_dt=0.01)
        )
        self.assertEqual(associated.reference.count, 2)
        self.assertEqual(np.unique(associated.reference.timestamps).shape[0], 2)
        np.testing.assert_allclose(np.sort(associated.source_time_error_s), [0.0, 0.004])

    def test_exact_does_not_accept_nearby_stamp(self) -> None:
        reference = trajectory(np.array([0.0, 1.0]), np.zeros((2, 3)))
        estimate = trajectory(np.array([1.0e-6, 1.0]), np.zeros((2, 3)))
        associated = associate_trajectories(
            reference,
            estimate,
            base_frame_config(association="exact", exact_tolerance=1.0e-9),
        )
        self.assertEqual(associated.reference.count, 1)

    def test_rejects_body_frame_mismatch_before_association(self) -> None:
        reference = trajectory(np.array([0.0, 1.0]), np.zeros((2, 3)))
        # No timestamps overlap, so an association attempt would return an
        # empty result.  The frame error must be raised first instead.
        estimate = trajectory(np.array([10.0, 11.0]), np.zeros((2, 3)))
        config = EvaluationConfig(
            reference_body_frame="base_link",
            estimate_body_frame="imu_sensor_frame",
            association="exact",
        )
        with self.assertRaisesRegex(EvaluationError, "body-frame mismatch"):
            associate_trajectories(reference, estimate, config)


class CommandLineApiTest(unittest.TestCase):
    def test_both_body_frame_inputs_are_required(self) -> None:
        parser = build_argument_parser()
        arguments = parser.parse_args(["reference.tum", "estimate.tum"])
        with self.assertRaisesRegex(EvaluationError, "manual mode requires"):
            _config_from_arguments(arguments)

        arguments = parser.parse_args(
            [
                "reference.tum",
                "estimate.tum",
                "--reference-body-frame",
                "base_link",
                "--estimate-body-frame",
                "base_link",
            ]
        )
        self.assertEqual(arguments.reference_body_frame, "base_link")
        self.assertEqual(arguments.estimate_body_frame, "base_link")
        self.assertEqual(_config_from_arguments(arguments).association, "interpolate")

    def test_scenario_mode_rejects_protocol_override(self) -> None:
        root = Path(__file__).parent.parent
        parser = build_argument_parser()
        arguments = parser.parse_args(
            [
                str(root / "bags/newer-college/gt/tum/gt-nc-quad-easy.csv"),
                "trajectory_base.tum",
                "--scenario",
                str(root / "benchmarks/scenarios/newer_college_quad_easy.yaml"),
                "--profile",
                "internal_interpolated",
                "--min-coverage",
                "0.5",
            ]
        )
        with self.assertRaisesRegex(EvaluationError, "forbids protocol overrides"):
            _config_from_arguments(arguments)


class ScenarioProfileTest(unittest.TestCase):
    def test_quad_easy_fixed_and_published_profiles_are_distinct(self) -> None:
        root = Path(__file__).parent.parent
        scenario = root / "benchmarks/scenarios/newer_college_quad_easy.yaml"
        ground_truth = root / "bags/newer-college/gt/tum/gt-nc-quad-easy.csv"
        internal = load_scenario_evaluation_config(
            scenario, "internal_interpolated", ground_truth, "local"
        )
        self.assertEqual(internal.coverage_protocol, FIXED_COLD_START_PROTOCOL)
        self.assertEqual(internal.startup_allowance_s, 2.2)
        self.assertEqual(internal.maximum_startup_delay_s, 2.2)
        self.assertEqual(internal.maximum_end_loss_s, 0.20)
        self.assertEqual(internal.min_coverage, 0.99)
        self.assertFalse(internal.publication_eligible)

        published = load_scenario_evaluation_config(
            scenario, "published_nearest_10ms", ground_truth, "final-global"
        )
        self.assertEqual(published.coverage_protocol, FULL_REFERENCE_PROTOCOL)
        self.assertEqual(published.association, "nearest")
        self.assertEqual(published.max_dt, 0.01)
        self.assertTrue(published.publication_eligible)

    def test_all_native_ros2_scenarios_use_the_same_versioned_cold_start_policy(self) -> None:
        root = Path(__file__).parent.parent
        cases = (
            ("newer_college_quad_easy.yaml", "gt-nc-quad-easy.csv"),
            ("newer_college_quad_hard.yaml", "gt-nc-quad-hard.csv"),
            ("newer_college_park_partial.yaml", "gt-nc-park.csv"),
        )
        for scenario_name, ground_truth_name in cases:
            with self.subTest(scenario=scenario_name):
                config = load_scenario_evaluation_config(
                    root / "benchmarks/scenarios" / scenario_name,
                    "internal_interpolated",
                    root / "bags/newer-college/gt/tum" / ground_truth_name,
                    "local",
                )
                self.assertEqual(config.coverage_protocol, FIXED_COLD_START_PROTOCOL)
                self.assertEqual(config.startup_allowance_s, 2.2)
                self.assertEqual(config.maximum_startup_delay_s, 2.2)
                self.assertEqual(config.maximum_end_loss_s, 0.20)
                self.assertEqual(config.min_coverage, 0.99)

    def test_publication_profile_cannot_use_fixed_cold_start_protocol(self) -> None:
        with self.assertRaisesRegex(EvaluationError, "cannot be publication eligible"):
            base_frame_config(
                association="interpolate",
                coverage_protocol=FIXED_COLD_START_PROTOCOL,
                startup_allowance_s=2.2,
                maximum_startup_delay_s=2.2,
                maximum_end_loss_s=0.2,
                publication_eligible=True,
            ).validate()

    def test_duplicate_scenario_key_is_rejected(self) -> None:
        root = Path(__file__).parent.parent
        source = (
            root / "benchmarks/scenarios/newer_college_quad_easy.yaml"
        ).read_text(encoding="utf-8")
        ambiguous = source.replace(
            "    purpose: meridian_development_and_acceptance\n",
            "    purpose: meridian_development_and_acceptance\n"
            "    purpose: silently_replaced_value\n",
            1,
        )
        temporary = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        temporary.write(ambiguous)
        temporary.close()
        scenario = Path(temporary.name)
        self.addCleanup(scenario.unlink, missing_ok=True)
        with self.assertRaisesRegex(EvaluationError, "duplicate key"):
            load_scenario_evaluation_config(
                scenario,
                "internal_interpolated",
                root / "bags/newer-college/gt/tum/gt-nc-quad-easy.csv",
                "local",
            )


class MetricTest(unittest.TestCase):
    def curved_reference(self, count: int = 81) -> Trajectory:
        stamps = np.linspace(0.0, 40.0, count)
        theta = stamps / 10.0
        positions = np.column_stack((5.0 * np.cos(theta), 5.0 * np.sin(theta), 0.1 * stamps))
        return trajectory(stamps, positions, theta + 0.5 * math.pi)

    def test_rigid_frame_change_has_zero_ate_and_rotation_error(self) -> None:
        reference = self.curved_reference()
        angle = 0.7
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle), 0.0],
             [math.sin(angle), math.cos(angle), 0.0],
             [0.0, 0.0, 1.0]]
        )
        translation = np.array([3.0, -2.0, 0.8])
        # Make an estimate in another world frame: p_ref = R*p_est+t.
        estimate_positions = (reference.positions - translation) @ rotation
        estimate_yaw = reference.timestamps / 10.0 + 0.5 * math.pi - angle
        estimate = trajectory(reference.timestamps, estimate_positions, estimate_yaw)
        report = evaluate_trajectories(
            reference,
            estimate,
            base_frame_config(
                association="exact",
                min_coverage=1.0,
                max_output_gap=1.0,
                rpe_distance_m=10.0,
                track_label="final-global",
            ),
        )
        self.assertLess(report["ate_translation_m"]["rmse"], 1.0e-12)
        self.assertLess(report["absolute_rotation_deg"]["rmse"], 1.0e-6)
        self.assertLess(report["rpe"]["translation_m"]["rmse"], 1.0e-12)
        self.assertTrue(report["quality"]["passed"])
        self.assertEqual(report["schema"], "meridian.trajectory_evaluation.v3")
        self.assertEqual(report["protocol"]["reference_body_frame"], "base_link")
        self.assertEqual(report["protocol"]["estimate_body_frame"], "base_link")

    def test_ten_meter_rpe_reports_scale_drift_without_scale_alignment(self) -> None:
        stamps = np.arange(0.0, 41.0, 1.0)
        # A slightly curved path keeps the SE(3) alignment observable.
        reference_positions = np.column_stack(
            (stamps, 0.02 * stamps * stamps, np.zeros_like(stamps))
        )
        estimate_positions = 1.1 * reference_positions
        reference = trajectory(stamps, reference_positions)
        estimate = trajectory(stamps, estimate_positions)
        report = evaluate_trajectories(
            reference,
            estimate,
            base_frame_config(
                association="exact",
                min_coverage=1.0,
                max_output_gap=2.0,
                rpe_distance_m=10.0,
                track_label="local",
            ),
        )
        drift = report["rpe"]["translation_drift_percent"]["mean"]
        self.assertGreater(drift, 9.0)
        self.assertLess(drift, 11.0)
        self.assertAlmostEqual(report["protocol"]["alignment_scale"], 1.0)

    def test_partial_track_and_output_gap_fail_quality_gate(self) -> None:
        stamps = np.arange(0.0, 11.0, 1.0)
        reference = trajectory(stamps, np.column_stack((stamps, stamps * 0.1, stamps * 0.0)))
        estimate_times = np.array([2.0, 3.0, 4.0, 8.0])
        estimate = trajectory(
            estimate_times,
            np.column_stack((estimate_times, estimate_times * 0.1, estimate_times * 0.0)),
        )
        report = evaluate_trajectories(
            reference,
            estimate,
            base_frame_config(
                association="exact",
                min_coverage=0.9,
                max_output_gap=2.0,
                track_label="local",
            ),
        )
        self.assertFalse(report["quality"]["passed"])
        self.assertEqual(report["estimate_health"]["gap_count"], 1)
        self.assertTrue(any("coverage" in item for item in report["quality"]["failures"]))

    def test_pose_discontinuity_is_reported_as_reset_candidate(self) -> None:
        stamps = np.arange(0.0, 6.0, 1.0)
        positions = np.column_stack((stamps, np.zeros_like(stamps), np.zeros_like(stamps)))
        positions[3:] += np.array([100.0, 0.0, 0.0])
        reference = trajectory(stamps, positions)
        estimate = trajectory(stamps, positions)
        report = evaluate_trajectories(
            reference,
            estimate,
            base_frame_config(
                association="exact",
                min_coverage=1.0,
                max_output_gap=2.0,
                max_linear_speed=30.0,
                track_label="local",
            ),
        )
        self.assertFalse(report["quality"]["passed"])
        self.assertEqual(report["estimate_health"]["reset_count"], 1)

    def test_fixed_window_cannot_hide_a_late_start(self) -> None:
        reference_times = np.arange(0.0, 100.0001, 0.1)
        reference = trajectory(
            reference_times,
            np.column_stack((reference_times, 0.01 * reference_times**2, 0.0 * reference_times)),
        )
        estimate_times = reference_times[30:]
        estimate = trajectory(
            estimate_times,
            np.column_stack((estimate_times, 0.01 * estimate_times**2, 0.0 * estimate_times)),
        )
        report = evaluate_trajectories(
            reference,
            estimate,
            base_frame_config(
                association="exact",
                coverage_protocol=FIXED_COLD_START_PROTOCOL,
                startup_allowance_s=2.2,
                maximum_startup_delay_s=2.2,
                maximum_end_loss_s=0.2,
                min_coverage=0.99,
                max_output_gap=0.2,
                track_label="local",
            ),
        )
        self.assertGreaterEqual(report["association"]["reference_pose_coverage_ratio"], 0.99)
        self.assertGreaterEqual(report["association"]["time_coverage_ratio"], 0.99)
        self.assertAlmostEqual(
            report["protocol"]["reference_window"]["start_timestamp"], 2.2
        )
        self.assertAlmostEqual(report["availability"]["startup_delay_s"], 3.0)
        self.assertFalse(report["availability"]["startup_delay_passed"])
        self.assertFalse(report["quality"]["passed"])
        self.assertTrue(any("startup delay" in item for item in report["quality"]["failures"]))

    def test_end_loss_is_independent_of_fixed_window_coverage(self) -> None:
        reference_times = np.arange(0.0, 100.0001, 0.1)
        reference = trajectory(
            reference_times,
            np.column_stack((reference_times, 0.01 * reference_times**2, 0.0 * reference_times)),
        )
        estimate_times = reference_times[20:-3]
        estimate = trajectory(
            estimate_times,
            np.column_stack((estimate_times, 0.01 * estimate_times**2, 0.0 * estimate_times)),
        )
        report = evaluate_trajectories(
            reference,
            estimate,
            base_frame_config(
                association="exact",
                coverage_protocol=FIXED_COLD_START_PROTOCOL,
                startup_allowance_s=2.2,
                maximum_startup_delay_s=2.2,
                maximum_end_loss_s=0.2,
                min_coverage=0.99,
                max_output_gap=0.2,
                track_label="local",
            ),
        )
        self.assertGreaterEqual(report["association"]["reference_pose_coverage_ratio"], 0.99)
        self.assertAlmostEqual(report["availability"]["end_loss_s"], 0.3)
        self.assertFalse(report["availability"]["end_loss_passed"])
        self.assertFalse(report["quality"]["passed"])
        self.assertTrue(any("end loss" in item for item in report["quality"]["failures"]))

    def test_theoretical_quad_easy_whole_pose_coverage_remains_visible(self) -> None:
        # Preserve the audited QE GT duration and the first GT stamp on/after
        # the earliest possible two-second motion-initialization commit.
        reference_duration_s = 198.682546854
        first_matched_delay_s = 2.000617027
        reference_times = np.linspace(0.0, reference_duration_s, 1988)
        reference_times[20] = first_matched_delay_s
        reference = trajectory(
            reference_times,
            np.column_stack((reference_times, 0.001 * reference_times**2, 0.0 * reference_times)),
        )
        # The native QE timing audit admits the first final initialization
        # state at GT index 20.  The exact whole-reference pose coverage is
        # therefore 1968/1988 = 0.9899396378..., below 0.99 by construction.
        estimate_times = reference_times[20:]
        estimate = trajectory(
            estimate_times,
            np.column_stack((estimate_times, 0.001 * estimate_times**2, 0.0 * estimate_times)),
        )
        report = evaluate_trajectories(
            reference,
            estimate,
            base_frame_config(
                association="exact",
                coverage_protocol=FIXED_COLD_START_PROTOCOL,
                startup_allowance_s=2.2,
                maximum_startup_delay_s=2.2,
                maximum_end_loss_s=0.2,
                min_coverage=0.99,
                max_output_gap=0.2,
                track_label="local",
            ),
        )
        theoretical = 1968.0 / 1988.0
        self.assertAlmostEqual(theoretical, 0.9899396378269618)
        self.assertAlmostEqual(
            report["whole_reference_association"]["reference_pose_coverage_ratio"],
            theoretical,
        )
        theoretical_time = (
            reference_duration_s - first_matched_delay_s
        ) / reference_duration_s
        self.assertAlmostEqual(theoretical_time, 0.9899305849509134)
        self.assertAlmostEqual(
            report["whole_reference_association"]["time_coverage_ratio"],
            theoretical_time,
        )
        self.assertLess(
            report["whole_reference_association"]["reference_pose_coverage_ratio"], 0.99
        )
        self.assertTrue(report["quality"]["passed"])


if __name__ == "__main__":
    unittest.main()
