// Copyright (c) OpenMMLab. All rights reserved.

#include "pose_detector.h"

#include "common.h"
#include "core/logger.h"

namespace mmdeploy {

class PyPoseDedector {
 public:
  PyPoseDedector(const char *model_path, const char *device_name, int device_id) {
    MMDEPLOY_INFO("{}, {}, {}", model_path, device_name, device_id);
    auto status =
        mmdeploy_pose_detector_create_by_path(model_path, device_name, device_id, &handle_);
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to create pose_detedtor");
    }
  }
  py::list Apply(const std::vector<PyImage> &imgs, const std::vector<std::vector<float>> &_boxes) {
    std::vector<mm_mat_t> mats;
    std::vector<mm_rect_t> boxes;
    mats.reserve(imgs.size());
    for (const auto &img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }
    for (const auto &_box : _boxes) {
      mm_rect_t box = {_box[0], _box[1], _box[2], _box[3]};
      boxes.push_back(box);
    }
    mm_pose_detect_t *detection{};
    int num_box = boxes.size();
    auto status = mmdeploy_pose_detector_apply_bbox(handle_, mats.data(), (int)mats.size(),
                                                    boxes.data(), &num_box, &detection);
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to apply pose_detector, code: " + std::to_string(status));
    }
    auto output = py::list{};
    auto result = detection;
    for (int i = 0; i < mats.size(); i++) {
      int n_point = result->length;
      auto pred = py::array_t<float>({1, n_point, 3});
      auto dst = pred.mutable_data();
      for (int j = 0; j < n_point; j++) {
        dst[0] = result->point[j].x;
        dst[1] = result->point[j].y;
        dst[2] = result->score[j];
        dst += 3;
      }
      output.append(std::move(pred));
      result++;
    }
    mmdeploy_pose_detector_release_result(detection, (int)mats.size());
    return output;
  }
  ~PyPoseDedector() {
    mmdeploy_pose_detector_destroy(handle_);
    handle_ = {};
  }

 private:
  mm_handle_t handle_{};
};

static void register_python_pose_detector(py::module &m) {
  py::class_<PyPoseDedector>(m, "PoseDetector")
      .def(py::init([](const char *model_path, const char *device_name, int device_id) {
        return std::make_unique<PyPoseDedector>(model_path, device_name, device_id);
      }))
      .def("__call__", &PyPoseDedector::Apply);
}

class PythonPoseDetectorRegisterer {
 public:
  PythonPoseDetectorRegisterer() {
    gPythonBindings().emplace("pose_detector", register_python_pose_detector);
  }
};

static PythonPoseDetectorRegisterer python_pose_detector_registerer;

}  // namespace mmdeploy
