#include "ct/keyframe_finalizer.hpp"

#include <ceres/problem.h>

#include <utility>

#ifdef __linux__
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "ct/pose_marginal.hpp"
#include "ct/window_problem.hpp"
#include "meridian/common/cov_reorder.hpp"
#include "meridian/debug/telemetry.hpp"
#include "meridian/debug/telemetry_keys.hpp"

namespace meridian {

KeyframeFinalizer::KeyframeFinalizer(Sink sink, TelemetrySink* telemetry, std::size_t capacity)
    : sink_(std::move(sink)), telemetry_(telemetry), queue_(capacity == 0 ? 1 : capacity) {}

KeyframeFinalizer::~KeyframeFinalizer() {
  stop();
}

void KeyframeFinalizer::start() {
  if (started_.exchange(true)) {
    return;
  }
  thread_ = std::thread([this] { worker_loop(); });
}

void KeyframeFinalizer::stop() {
  if (!started_.exchange(false)) {
    // Never started: nothing to drain or join.
    return;
  }
  queue_.close();
  if (thread_.joinable()) {
    thread_.join();
  }
}

bool KeyframeFinalizer::submit(KeyframeJob&& job) {
  {
    std::lock_guard<std::mutex> lock(drain_mtx_);
    ++submitted_;
  }
  return queue_.push_blocking(std::move(job));
}

void KeyframeFinalizer::worker_loop() {
#ifdef __linux__
  // Each job has ~1 s of slack (the next keyframe), while the estimator's solve and
  // association threads live on a per-sweep deadline. Demote this thread to the
  // lowest scheduling weight so under contention it yields virtually every cycle to
  // them and runs on cores they leave idle; uncontended it still runs at full speed.
  setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 19);
#endif
  KeyframeJob job;
  while (queue_.pop(job)) {
    finalize(std::move(job));
  }
}

bool KeyframeFinalizer::poseMarginal(KeyframeJob& job, Eigen::Matrix<double, 6, 6>* cov) {
  // No captured window (a fallback-only job): the marginal cannot be rebuilt, so the
  // caller uses the job's LiDAR-only fallback cov.
  if (!job.spline || !job.bias) {
    return false;
  }
  // Each problem owns its cost functions/manifolds and frees them on destruction. The
  // job's spline/bias/gravity/prior are clones, so wiring the inputs to them and
  // rebuilding reproduces the final outer round's problem on worker-owned storage.
  job.inputs.spline = job.spline.get();
  job.inputs.bias = job.bias.get();
  job.inputs.gravity = &job.gravity;
  job.inputs.prior = job.prior.get();
  job.inputs.capped_hits = &job.capped_hits;
  job.inputs.imu = &job.imu;

  ceres::Problem problem;
  ct::WindowExposureScratch expo(job.inputs.visual_cfg);
  ct::WindowExposureScratch* expo_ptr = job.inputs.visual_patches.empty() ? nullptr : &expo;
  ct::buildWindowProblem(problem, job.inputs, expo_ptr);

  return ct::poseMarginalFromProblem(problem, *job.spline, job.stamp, cov);
}

void KeyframeFinalizer::finalize(KeyframeJob&& job) {
  const auto t_start = Clock::now();

  // Recover the window-posterior pose marginal on the rebuilt clone. On failure the
  // hot thread already packed the LiDAR-only fallback into the job's pose marginal, so
  // we keep that and only the chained constraint_cov below changes.
  Eigen::Matrix<double, 6, 6> pose_cov;
  const bool ok = poseMarginal(job, &pose_cov);
  if (!ok) {
    // Rank-deficient rebuilt posterior: use the LiDAR-only marginal the hot thread
    // packed into the job, matching the hot thread's own fallback choice.
    pose_cov = job.fallback_pose_cov;
  }

  // Chain the relative-edge covariance. The worker owns the chain: FIFO order makes the
  // previous finalized keyframe the immediate predecessor. An AbsolutePrior break (first
  // keyframe / gap reseed) resets the chain to this keyframe's absolute marginal.
  if (job.absolute_prior || !have_prev_) {
    job.pkt.constraint_cov.form = GaussianBlock<6>::Form::Covariance;
    job.pkt.constraint_cov.M = reorderTransRotToRotTrans(pose_cov);
  } else {
    // Conservative relative cov: the sum of the two absolute pose marginals upper-bounds
    // the true relative marginal (which subtracts their cross-covariance).
    const Eigen::Matrix<double, 6, 6> rel_cov = pose_cov + prev_kf_pose_cov_;
    job.pkt.constraint_cov.form = GaussianBlock<6>::Form::Covariance;
    job.pkt.constraint_cov.M = reorderTransRotToRotTrans(rel_cov);
  }

  prev_kf_pose_cov_ = pose_cov;
  have_prev_ = true;

  {
    std::lock_guard<std::mutex> lock(cov_mtx_);
    last_pose_cov_ = pose_cov;
    have_pose_cov_ = true;
  }

  if (telemetry_ != nullptr && telemetry_->enabled(keys::stage::CtPosecovWorker)) {
    telemetry_->scalar(keys::stage::CtPosecovWorker, ms_since(t_start), job.stamp);
  }

  if (sink_) {
    sink_(std::move(job.pkt));
  }

  {
    std::lock_guard<std::mutex> lock(drain_mtx_);
    ++finalized_;
  }
  drain_cv_.notify_all();
}

void KeyframeFinalizer::drain_for_test() {
  std::unique_lock<std::mutex> lock(drain_mtx_);
  drain_cv_.wait(lock, [this] { return finalized_ >= submitted_; });
}

Eigen::Matrix<double, 6, 6> KeyframeFinalizer::last_pose_cov() const {
  std::lock_guard<std::mutex> lock(cov_mtx_);
  return last_pose_cov_;
}

bool KeyframeFinalizer::have_pose_cov() const {
  std::lock_guard<std::mutex> lock(cov_mtx_);
  return have_pose_cov_;
}

}  // namespace meridian
