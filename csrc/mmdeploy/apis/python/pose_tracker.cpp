// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/pose_tracker.hpp"

#include "common.h"
#include "mmdeploy/common.hpp"

namespace mmdeploy::python {

namespace {

std::vector<py::tuple> Apply(mmdeploy::PoseTracker* self,
                             const std::vector<mmdeploy::PoseTracker::State*>& _states,
                             const std::vector<PyImage>& _frames, std::vector<int> detect) {
  std::vector<mmdeploy_pose_tracker_state_t> tmp;
  for (const auto& s : _states) {
    tmp.push_back(static_cast<mmdeploy_pose_tracker_state_t>(*s));
  }
  mmdeploy::Span states(reinterpret_cast<mmdeploy::PoseTracker::State*>(tmp.data()), tmp.size());
  std::vector<mmdeploy::Mat> frames;
  for (const auto& f : _frames) {
    frames.emplace_back(GetMat(f));
  }
  if (detect.empty()) {
    detect.resize(frames.size(), -1);
  }
  assert(states.size() == frames.size());
  assert(states.size() == detect.size());
  auto results = self->Apply(states, frames, detect);
  std::vector<py::tuple> batch_ret;
  batch_ret.reserve(frames.size());
  for (const auto& rs : results) {
    py::array_t<float> keypoints(
        {static_cast<int>(rs.size()), rs.size() > 0 ? rs[0].keypoint_count : 0, 3});
    py::array_t<float> bboxes({static_cast<int>(rs.size()), 4});
    py::array_t<uint32_t> track_ids(static_cast<int>(rs.size()));
    auto kpts_ptr = keypoints.mutable_data();
    auto bbox_ptr = bboxes.mutable_data();
    auto track_id_ptr = track_ids.mutable_data();
    for (const auto& r : rs) {
      for (int i = 0; i < r.keypoint_count; ++i) {
        kpts_ptr[0] = r.keypoints[i].x;
        kpts_ptr[1] = r.keypoints[i].y;
        kpts_ptr[2] = r.scores[i];
        kpts_ptr += 3;
      }
      {
        auto tmp_bbox = (std::array<float, 4>&)r.bbox;
        bbox_ptr[0] = tmp_bbox[0];
        bbox_ptr[1] = tmp_bbox[1];
        bbox_ptr[2] = tmp_bbox[2];
        bbox_ptr[3] = tmp_bbox[3];
        bbox_ptr += 4;
      }
      *track_id_ptr++ = r.target_id;
    }
    batch_ret.push_back(
        py::make_tuple(std::move(keypoints), std::move(bboxes), std::move(track_ids)));
  }
  return batch_ret;
}

template <typename T, size_t N>
void Copy(const py::handle& h, T (&a)[N]) {
  auto array = h.cast<py::array_t<float>>();
  assert(array.size() == N);
  auto data = array.data();
  for (int i = 0; i < N; ++i) {
    a[i] = data[i];
  }
}

void Parse(const py::dict& dict, PoseTracker::Params& params, py::array_t<float>& sigmas) {
  for (const auto& [_name, value] : dict) {
    auto name = _name.cast<std::string>();
    if (name == "det_interval") {
      params->det_interval = value.cast<int32_t>();
    } else if (name == "det_label") {
      params->det_label = value.cast<int32_t>();
    } else if (name == "det_thr") {
      params->det_thr = value.cast<float>();
    } else if (name == "det_min_bbox_size") {
      params->det_min_bbox_size = value.cast<float>();
    } else if (name == "det_nms_thr") {
      params->det_nms_thr = value.cast<float>();
    } else if (name == "pose_max_num_bboxes") {
      params->pose_max_num_bboxes = value.cast<int32_t>();
    } else if (name == "pose_min_keypoints") {
      params->pose_min_keypoints = value.cast<int32_t>();
    } else if (name == "pose_min_bbox_size") {
      params->pose_min_bbox_size = value.cast<float>();
    } else if (name == "pose_nms_thr") {
      params->pose_nms_thr = value.cast<float>();
    } else if (name == "track_kpt_thr") {
      params->pose_kpt_thr = value.cast<float>();
    } else if (name == "track_iou_thr") {
      params->track_iou_thr = value.cast<float>();
    } else if (name == "pose_bbox_scale") {
      params->pose_bbox_scale = value.cast<float>();
    } else if (name == "track_max_missing") {
      params->track_max_missing = value.cast<float>();
    } else if (name == "track_history_size") {
      params->track_history_size = value.cast<int32_t>();
    } else if (name == "keypoint_sigmas") {
      sigmas = value.cast<py::array_t<float>>();
      params->keypoint_sigmas = const_cast<float*>(sigmas.data());
      params->keypoint_sigmas_size = sigmas.size();
    } else if (name == "std_weight_position") {
      params->std_weight_position = value.cast<float>();
    } else if (name == "std_weight_velocity") {
      params->std_weight_velocity = value.cast<float>();
    } else if (name == "smooth_params") {
      Copy(value, params->smooth_params);
    } else {
      MMDEPLOY_ERROR("unused argument: {}", name);
    }
  }
}

}  // namespace

static PythonBindingRegisterer register_pose_tracker{[](py::module& m) {
  py::class_<mmdeploy::PoseTracker::State>(m, "PoseTracker.State");
  py::class_<mmdeploy::PoseTracker>(m, "PoseTracker")
      .def(py::init([](const char* det_model_path, const char* pose_model_path,
                       const char* device_name, int device_id) {
             return mmdeploy::PoseTracker(
                 mmdeploy::Model(det_model_path), mmdeploy::Model(pose_model_path),
                 mmdeploy::Context(mmdeploy::Device(device_name, device_id)));
           }),
           py::arg("det_model"), py::arg("pose_model"), py::arg("device_name"),
           py::arg("device_id") = 0)
      .def(
          "__call__",
          [](mmdeploy::PoseTracker* self, mmdeploy::PoseTracker::State* state, const PyImage& img,
             int detect) { return Apply(self, {state}, {img}, {detect})[0]; },
          py::arg("state"), py::arg("frame"), py::arg("detect") = -1)
      .def("batch", &Apply, py::arg("states"), py::arg("frames"),
           py::arg("detects") = std::vector<int>{})
      .def("create_state", [](mmdeploy::PoseTracker* self, const py::kwargs& kwargs) {
        PoseTracker::Params params;
        py::array_t<float> sigmas;
        if (kwargs) {
          Parse(kwargs, params, sigmas);
        }
        return self->CreateState(params);
      });
}};

}  // namespace mmdeploy::python
