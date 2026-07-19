#include <cstdint>
#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

#include "meridian/apps/bag_localize.hpp"

namespace {

void printUsage(std::ostream& stream) {
  stream << "Usage: meridian_bag_localize --bag PATH --calib PATH "
            "--collection 1|2 --sensor-mode imu|lidar-imu|full "
            "[--initialization dynamic|static|supervised-auto] "
            "--output DIRECTORY --max-events N --max-bag-messages N "
            "--replay-mode scheduled|unpaced [--playback-rate RATE] "
            "[--queue-count-capacity N] "
            "[--queue-byte-capacity N] [--timing-window-capacity N] "
            "[--maximum-lidar-targets N] "
            "[--visual-graph enabled|disabled]\n\n"
         << "Runs Newer College ROS 2 localization in scheduled or unpaced mode. "
            "Scheduled replay requires a finite positive RATE: 1 is recorded real time, "
            "and values above 1 accelerate replay. Unpaced replay forbids RATE and "
            "measures algorithm throughput without wall-clock pacing. Queue count/byte "
            "and rolling timing capacities have deterministic defaults and every "
            "effective value is recorded in run_report.json. In scheduled mode, a "
            "nonempty terminal queue, producer/consumer "
            "count mismatch, reject, drop, or skip fails the benchmark. Sensor selection "
            "is applied before bag deserialization; "
            "dynamic initialization is the default. Selecting static is an explicit "
            "operator zero-motion assertion and forbids dynamic fallback; supervised-auto "
            "verifies the same assertion but may fall back to LiDAR--IMU initialization. "
            "--maximum-lidar-targets accepts 1 or 2 and changes only the bounded pose-aware "
            "target count for registration benchmarks. "
            "--visual-graph is valid only with --sensor-mode full. ROS 1 input is "
            "not enabled.\n";
}

void printOptionalNanoseconds(std::ostream& stream, const std::optional<std::int64_t>& value) {
  if (value) {
    stream << *value;
  } else {
    stream << "null";
  }
}

[[nodiscard]] std::string_view replayModeName(
    meridian::apps::BagLocalizationReplayMode mode) noexcept {
  switch (mode) {
    case meridian::apps::BagLocalizationReplayMode::Scheduled:
      return "scheduled";
    case meridian::apps::BagLocalizationReplayMode::Unpaced:
      return "unpaced";
  }
  return "unknown";
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string_view> raw_arguments;
  raw_arguments.reserve(argc > 1 ? static_cast<std::size_t>(argc - 1) : 0U);
  for (int index = 1; index < argc; ++index) {
    raw_arguments.emplace_back(argv[index]);
  }

  auto arguments = meridian::apps::parseBagLocalizationArguments(raw_arguments);
  if (!arguments) {
    std::cerr << "meridian_bag_localize: " << arguments.error().detail << '\n';
    printUsage(std::cerr);
    return 2;
  }
  if (arguments.value().help) {
    printUsage(std::cout);
    return 0;
  }

  const meridian::apps::BagLocalizationOutcome outcome =
      meridian::apps::runBagLocalization(arguments.value().options);
  std::cout << "run_report=" << outcome.run_report << '\n'
            << "trajectory_imu=" << outcome.imu_trajectory << '\n'
            << "trajectory_base=" << outcome.base_trajectory << '\n'
            << "navigation_diagnostics=" << outcome.navigation_diagnostics << '\n'
            << "states_written=" << outcome.states_written << '\n';
  if (outcome.runtime) {
    const auto& runtime = *outcome.runtime;
    std::cout << "replay_mode=" << replayModeName(runtime.replay_mode) << '\n'
              << "playback_rate=" << runtime.playback_rate << '\n'
              << "queue_terminal_count=" << runtime.queue_terminal_count << '\n'
              << "queue_terminal_bytes=" << runtime.queue_terminal_bytes << '\n'
              << "queue_maximum_count=" << runtime.queue_maximum_count << '/'
              << runtime.queue_count_capacity << '\n'
              << "queue_maximum_bytes=" << runtime.queue_maximum_bytes << '/'
              << runtime.queue_byte_capacity << '\n'
              << "timing_window_capacity=" << runtime.timing_window_capacity << '\n'
              << "queue_maximum_oldest_age_ns=";
    printOptionalNanoseconds(std::cout, runtime.queue_maximum_oldest_age_ns);
    std::cout << '\n'
              << "queue_rejected=" << runtime.queue_rejected << '\n'
              << "queue_dropped=" << runtime.queue_dropped << '\n'
              << "queue_skipped=" << runtime.queue_skipped << '\n'
              << "producer_total_wall_ns=";
    printOptionalNanoseconds(std::cout, runtime.producer_total_wall_ns);
    std::cout << '\n' << "producer_p95_wall_ns=";
    printOptionalNanoseconds(std::cout, runtime.producer_p95_wall_ns);
    std::cout << '\n' << "consumer_total_wall_ns=";
    printOptionalNanoseconds(std::cout, runtime.consumer_total_wall_ns);
    std::cout << '\n' << "consumer_p95_wall_ns=";
    printOptionalNanoseconds(std::cout, runtime.consumer_p95_wall_ns);
    std::cout << '\n' << "process_ready_p95_wall_ns=";
    printOptionalNanoseconds(std::cout, runtime.process_ready_p95_wall_ns);
    std::cout << '\n' << "report_output_p95_wall_ns=";
    printOptionalNanoseconds(std::cout, runtime.report_output_p95_wall_ns);
    std::cout << '\n';
  }
  if (!outcome.error.empty()) {
    std::cerr << "meridian_bag_localize: " << outcome.error << '\n';
  }
  return outcome.suggested_exit_code;
}
