// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_POSE_TRACKER_HPP
#define MMDEPLOY_POSE_TRACKER_HPP

#include "mmdeploy/common.hpp"
#include "mmdeploy/pose_tracker.h"

namespace mmdeploy {

namespace cxx {

class PoseTracker : public UniqueHandle<mmdeploy_pose_tracker_t> {
 public:
  using Result = Result_<mmdeploy_pose_tracker_target_t>;
  class State;
  class Params;

 public:
  /**
   * @brief Create pose tracker pipeline
   * @param detect object detection model
   * @param pose pose estimation model
   * @param context execution context
   */
  PoseTracker(const Model& detect, const Model& pose, const Context& context) {
    auto ec = mmdeploy_pose_tracker_create(detect, pose, context, &handle_);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
  }
  ~PoseTracker() {
    if (handle_) {
      mmdeploy_pose_tracker_destroy(handle_);
      handle_ = {};
    }
  }
  PoseTracker(PoseTracker&&) noexcept = default;

  /**
   * @brief Create a tracker state corresponds to a video stream
   * @param params params for creating the tracker state
   * @return created tracker state
   */
  State CreateState(const Params& params);

  /**
   * @brief Apply pose tracker pipeline
   * @param state tracker state
   * @param frame input video frame
   * @param detect control the use of detector
   *   -1: use params.det_interval, 0: don't use detector, 1: force use detector
   * @return
   */
  Result Apply(State& state, const Mat& frame, int detect = -1);

  /**
   * @brief batched version of Apply
   * @param states
   * @param frames
   * @param detects
   * @return
   */
  std::vector<Result> Apply(const Span<State>& states, const Span<const Mat>& frames,
                            const Span<const int>& detects = {});

 public:
  /**
   * see \ref mmdeploy/pose_tracker.h for detail
   */
  class Params : public UniqueHandle<mmdeploy_pose_tracker_param_t*> {
   public:
    explicit Params() {
      handle_ = new mmdeploy_pose_tracker_param_t{};
      mmdeploy_pose_tracker_default_params(handle_);
    }
    ~Params() {
      if (handle_) {
        delete handle_;
        handle_ = {};
      }
    }
  };

  class State : public UniqueHandle<mmdeploy_pose_tracker_state_t> {
   public:
    explicit State(mmdeploy_pose_tracker_t pipeline, const mmdeploy_pose_tracker_param_t* params) {
      auto ec = mmdeploy_pose_tracker_create_state(pipeline, params, &handle_);
      if (ec != MMDEPLOY_SUCCESS) {
        throw_exception(static_cast<ErrorCode>(ec));
      }
    }
    ~State() {
      if (handle_) {
        mmdeploy_pose_tracker_destroy_state(handle_);
        handle_ = {};
      }
    }
    State(State&&) noexcept = default;
  };
};

inline PoseTracker::State PoseTracker::CreateState(const PoseTracker::Params& params) {
  return State(handle_, static_cast<mmdeploy_pose_tracker_param_t*>(params));
}

inline std::vector<PoseTracker::Result> PoseTracker::Apply(const Span<State>& states,
                                                           const Span<const Mat>& frames,
                                                           const Span<const int32_t>& detects) {
  if (frames.empty()) {
    return {};
  }
  mmdeploy_pose_tracker_target_t* results{};
  int32_t* result_count{};

  auto ec = mmdeploy_pose_tracker_apply(
      handle_, reinterpret_cast<mmdeploy_pose_tracker_state_t*>(states.data()),
      reinterpret(frames.data()), detects.data(), static_cast<int32_t>(frames.size()), &results,
      &result_count);
  if (ec != MMDEPLOY_SUCCESS) {
    throw_exception(static_cast<ErrorCode>(ec));
  }

  std::shared_ptr<mmdeploy_pose_tracker_target_t> data(
      results, [result_count, count = frames.size()](auto p) {
        mmdeploy_pose_tracker_release_result(p, result_count, count);
      });

  std::vector<Result> rets;
  rets.reserve(frames.size());

  size_t offset = 0;
  for (size_t i = 0; i < frames.size(); ++i) {
    offset += rets.emplace_back(offset, result_count[i], data).size();
  }

  return rets;
}

inline PoseTracker::Result PoseTracker::Apply(PoseTracker::State& state, const Mat& frame,
                                              int32_t detect) {
  return Apply(Span(&state, 1), Span{frame}, Span{detect})[0];
}

}  // namespace cxx

using cxx::PoseTracker;

}  // namespace mmdeploy

#endif  // MMDEPLOY_POSE_TRACKER_HPP
