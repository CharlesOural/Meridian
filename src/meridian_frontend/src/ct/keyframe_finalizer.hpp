#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "ct/marginalization.hpp"
#include "ct/residuals_imu.hpp"
#include "ct/residuals_lidar.hpp"
#include "ct/spline_window.hpp"
#include "ct/window_problem.hpp"
#include "meridian/common/bounded_queue.hpp"
#include "meridian/common/keyframe_packet.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"

namespace meridian {

class TelemetrySink;

// A self-contained unit of deferred keyframe work captured on the hot thread at
// keyframe-emit time. It owns clones of every input the FINAL outer round's window
// problem was built from, so the worker can rebuild a structurally identical
// ceres::Problem race-free after the live spline/problem have moved on. Building it
// must be cheap (clone knot values, copy the capped hits / IMU / patches / prior).
//
// The door is left open for future deferred keyframe fields (place-recognition
// descriptors, refined observability, heavy telemetry): add the captured input here and
// the processing step in KeyframeFinalizer::finalize.
struct KeyframeJob {
  // Worker-owned trajectory clones the rebuilt residuals point into. Held by value /
  // unique_ptr so the job is movable across the queue and outlives the live window.
  std::unique_ptr<SplineWindow> spline;
  std::unique_ptr<ct::BiasKnots> bias;
  ct::GravityBlock gravity{Eigen::Vector3d(0, 0, -1), 9.81};
  std::unique_ptr<ct::MarginalizationPrior> prior;  // null when no prior existed

  // The final round's build inputs, captured by value. The pointers inside (spline,
  // bias, gravity, prior, capped_hits, imu) are wired to this job's clones before the
  // worker calls buildWindowProblem.
  ct::WindowProblemInputs inputs;
  std::vector<ct::LidarHit> capped_hits;  // inputs.capped_hits points here
  std::vector<ImuSample> imu;             // inputs.imu points here

  Timestamp stamp = 0;  // the marginal evaluation time (sweep end)

  // LiDAR-only pose marginal computed on the hot thread, used as the absolute pose
  // marginal when the rebuilt window posterior is rank-deficient (mirrors the hot
  // thread's fall back from poseMarginalFromProblem to poseMarginal).
  Eigen::Matrix<double, 6, 6> fallback_pose_cov = Eigen::Matrix<double, 6, 6>::Identity();

  // The partial packet: everything except constraint_cov, filled on the hot thread.
  KeyframePacket pkt;
  // Chain decision captured on the hot thread (the worker owns the rel_cov chain, but
  // the AbsolutePrior-vs-RelativeBetween choice and the relative transform were decided
  // at emit time against the live previous-keyframe pose).
  bool absolute_prior = false;  // true => AbsolutePrior chain break (first KF / gap reseed)
};

// A single FIFO stage thread that pops KeyframeJob, rebuilds the window problem from the
// clone, recovers the pose marginal, chains it into the relative-edge covariance, packs
// constraint_cov, and forwards the finished packet to the keyframe sink. Owns the
// rel_cov chain state (prev_kf_pose_cov_) so FIFO order makes each job's predecessor the
// immediate one. The hot thread may also drive finalize() inline (queue-full fallback or
// the deterministic path) so a keyframe is never dropped and the chain has no holes.
class KeyframeFinalizer {
public:
  using Sink = std::function<void(KeyframePacket&&)>;

  // `sink` receives finished packets (thread-safe; it only enqueues). `telemetry` is
  // borrowed for the worker-side timing and may be null. Capacity is small; a full queue
  // triggers the inline fallback.
  KeyframeFinalizer(Sink sink, TelemetrySink* telemetry, std::size_t capacity = 4);
  ~KeyframeFinalizer();

  KeyframeFinalizer(const KeyframeFinalizer&) = delete;
  KeyframeFinalizer& operator=(const KeyframeFinalizer&) = delete;

  // Spawns the worker thread. Idempotent.
  void start();
  // Closes the queue, drains every in-flight job (each emitted with a filled cov), and
  // joins the worker. Idempotent; safe to call from the destructor.
  void stop();

  // Hands a job to the worker (FIFO). When the queue is full this BLOCKS the caller until
  // the worker frees a slot, so the chain is processed strictly in id order with no holes
  // and no drops. A full queue requires keyframe finalization to fall behind the ~1 Hz
  // keyframe rate, which a <100 ms finalize against a 4-deep queue does not do in steady
  // state; the bounded wait is the graceful overload fallback and is never worse than
  // today's fully-synchronous cost. Returns false only if the finalizer is stopping.
  bool submit(KeyframeJob&& job);

  // Finalizes one job synchronously on the caller's thread: rebuild -> marginal -> chain
  // -> reorder -> pack -> sink. Used by the worker, the deterministic path, and the
  // parity test. Advances the rel_cov chain state; call only from the single consumer
  // (the worker thread) or a single-threaded context (deterministic path / test).
  void finalize(KeyframeJob&& job);

  // The most recent pose marginal the worker (or inline finalize) produced, translation-
  // first [rho; phi], and whether one has been produced yet. The GNSS gate reads these
  // next sweep, tolerating the one-keyframe lag.
  Eigen::Matrix<double, 6, 6> last_pose_cov() const;
  bool have_pose_cov() const;

  // Blocks until every job submitted so far has been finalized (the queue is empty and
  // the worker is idle). Test helper so a test can assert on emitted packets without
  // racing the async worker; not used in production.
  void drain_for_test();

private:
  // Rebuild the cloned problem and recover the pose marginal (translation-first
  // [rho; phi]) via the same routine the hot thread used. Returns false when the joint
  // information is rank-deficient (the caller then leaves cov at the LiDAR-only fallback
  // the hot thread already packed into the job).
  bool poseMarginal(KeyframeJob& job, Eigen::Matrix<double, 6, 6>* cov);

  void worker_loop();

  Sink sink_;
  TelemetrySink* telemetry_ = nullptr;
  BoundedQueue<KeyframeJob> queue_;
  std::thread thread_;
  std::atomic<bool> started_{false};

  // rel_cov chain state, owned by the single consumer (the worker thread, or the calling
  // thread on the deterministic / test path). FIFO submission makes prev_* the immediate
  // predecessor keyframe. Never touched concurrently, so it needs no lock.
  bool have_prev_ = false;
  Eigen::Matrix<double, 6, 6> prev_kf_pose_cov_ = Eigen::Matrix<double, 6, 6>::Identity();

  mutable std::mutex cov_mtx_;
  Eigen::Matrix<double, 6, 6> last_pose_cov_ = Eigen::Matrix<double, 6, 6>::Identity();
  bool have_pose_cov_ = false;

  // Submitted-vs-finalized counters for the test drain. submitted_ is bumped on the
  // producer side, finalized_ on the consumer side; drain_for_test waits for equality.
  std::mutex drain_mtx_;
  std::condition_variable drain_cv_;
  std::uint64_t submitted_ = 0;
  std::uint64_t finalized_ = 0;
};

}  // namespace meridian
