// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "pose_tracker/common.h"
#include "pose_tracker/pose_tracker.h"

namespace mmdeploy {

MMDEPLOY_REGISTER_TYPE_ID(mmpose::_pose_tracker::Tracker, 0xcfe87980aa895d3a);

}

namespace mmdeploy::mmpose::_pose_tracker {

#define REGISTER_SIMPLE_MODULE(name, fn) \
  MMDEPLOY_REGISTER_FACTORY_FUNC(Module, (name, 0), [](const Value&) { return CreateTask(fn); });

Value Prepare(const Value& data, const Value& use_det, Value state) {
  auto& tracker = state.get_ref<Tracker&>();
  // set frame size for the first frame
  if (tracker.frame_id() == 0) {
    auto& frame = data["ori_img"].get_ref<const framework::Mat&>();
    tracker.SetFrameSize(frame.height(), frame.width());
  }
  // use_det is set to non-auto mode
  if (use_det.get<int>() != -1) {
    return use_det;
  }
  auto interval = tracker.params().det_interval;
  return interval > 0 && tracker.frame_id() % interval == 0;
}

REGISTER_SIMPLE_MODULE(pose_tracker::Prepare, Prepare);

std::tuple<Value, Value> ProcessBboxes(const Value& det_val, const Value& data,
                                       Value state) noexcept {
  auto& tracker = state.get_ref<Tracker&>();

  std::optional<Tracker::Detections> dets;

  if (det_val.is_array()) {  // has detections
    auto& [bboxes, scores, labels] = dets.emplace();
    for (const auto& det : det_val.array()) {
      bboxes.push_back(from_value<Bbox>(det["bbox"]));
      scores.push_back(det["score"].get<float>());
      labels.push_back(det["label_id"].get<int>());
    }
  }

  auto [bboxes, ids] = tracker.ProcessBboxes(dets);

  Value::Array bbox_array;
  Value track_ids_array;
  // attach bboxes to image data
  for (auto& bbox : bboxes) {
    cv::Rect rect(cv::Rect2f(cv::Point2f(bbox[0], bbox[1]), cv::Point2f(bbox[2], bbox[3])));
    bbox_array.push_back({
        {"img", data["img"]},                                 // img
        {"bbox", {rect.x, rect.y, rect.width, rect.height}},  // bbox
    });
  }

  track_ids_array = to_value(ids);
  return {std::move(bbox_array), std::move(track_ids_array)};
}
REGISTER_SIMPLE_MODULE(pose_tracker::ProcessBboxes, ProcessBboxes);

Value TrackStep(const Value& poses, const Value& track_indices, Value state) noexcept {
  assert(poses.is_array());
  vector<Points> keypoints;
  vector<Scores> scores;
  for (auto& output : poses.array()) {
    auto& k = keypoints.emplace_back();
    auto& s = scores.emplace_back();
    float avg = 0.f;
    for (auto& kpt : output["key_points"].array()) {
      k.emplace_back(kpt["bbox"][0].get<float>(), kpt["bbox"][1].get<float>());
      s.push_back(kpt["score"].get<float>());
      avg += s.back();
    }
  }
  vector<int64_t> track_ids;
  from_value(track_indices, track_ids);
  auto& tracker = state.get_ref<Tracker&>();
  tracker.TrackStep(keypoints, scores, track_ids);
  TrackerResult result;
  for (const auto& track : tracker.tracks()) {
    if (track->missing()) {
      continue;
    }
    vector<mmdeploy_point_t> kpts;
    kpts.reserve(track->smoothed_kpts().size());
    for (const auto& kpt : track->smoothed_kpts()) {
      kpts.push_back({kpt.x, kpt.y});
    }
    result.keypoints.push_back(std::move(kpts));
    result.scores.push_back(track->scores());
    auto& bbox = track->smoothed_bbox();
    result.bboxes.push_back({bbox[0], bbox[1], bbox[2], bbox[3]});
    result.track_ids.push_back(track->track_id());
  }
  result.pose_input_bboxes = tracker.pose_input_bboxes();
  result.pose_output_bboxes = tracker.pose_output_bboxes();
  return result;
}
REGISTER_SIMPLE_MODULE(pose_tracker::TrackStep, TrackStep);

// MSVC toolset v143 keeps ICEing when using a lambda here
static Value CreateTracker(mmdeploy_pose_tracker_param_t* param) {
  return make_pointer(Tracker{*param});
}
REGISTER_SIMPLE_MODULE(pose_tracker::Create, CreateTracker);

}  // namespace mmdeploy::mmpose::_pose_tracker
