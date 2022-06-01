// Copyright (c) OpenMMLab. All rights reserved.

#include "pose_detector.h"

#include <array>

#include "common.h"
#include "core/logger.h"

namespace mmdeploy {

using Rect = std::array<float, 4>;

class PyPoseDedector {
 public:
  PyPoseDedector(const char *model_path, const char *device_name, int device_id) {
    MMDEPLOY_INFO("{}, {}, {}", model_path, device_name, device_id);
    auto status =
        mmdeploy_pose_detector_create_by_path(model_path, device_name, device_id, &handle_);
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to create pose_detector");
    }
  }
  py::list Apply(const std::vector<PyImage> &imgs, const std::vector<std::vector<Rect>> &vboxes) {
    if (imgs.size() == 0 && vboxes.size() == 0) {
      return py::list{};
    }
    if (vboxes.size() != 0 && vboxes.size() != imgs.size()) {
      std::string error =
          fmt::format("imgs length not equal with vboxes [{} vs {}]", imgs.size(), vboxes.size());
      throw std::invalid_argument(error);
    }

    std::vector<mm_mat_t> mats;
    std::vector<mm_rect_t> boxes;
    std::vector<int> bbox_count;
    mats.reserve(imgs.size());
    for (const auto &img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }

    for (auto _boxes : vboxes) {
      for (auto _box : _boxes) {
        mm_rect_t box = {_box[0], _box[1], _box[2], _box[3]};
        boxes.push_back(box);
      }
      bbox_count.push_back(_boxes.size());
    }

    // full image
    if (vboxes.size() == 0) {
      for (int i = 0; i < mats.size(); i++) {
        mm_rect_t box = {0.f, 0.f, mats[i].width - 1, mats[i].height - 1};
        boxes.push_back(box);
        bbox_count.push_back(1);
      }
    }

    mm_pose_detect_t *detection{};
    auto status = mmdeploy_pose_detector_apply_bbox(handle_, mats.data(), (int)mats.size(),
                                                    boxes.data(), bbox_count.data(), &detection);
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to apply pose_detector, code: " + std::to_string(status));
    }

    auto output = py::list{};
    auto result = detection;
    for (int i = 0; i < mats.size(); i++) {
      if (bbox_count[i] == 0) {
        output.append(py::none());
        continue;
      }
      int n_point = result->length;
      auto pred = py::array_t<float>({bbox_count[i], n_point, 3});
      auto dst = pred.mutable_data();
      for (int j = 0; j < bbox_count[i]; j++) {
        for (int k = 0; k < n_point; k++) {
          dst[0] = result->point[k].x;
          dst[1] = result->point[k].y;
          dst[2] = result->score[k];
          dst += 3;
        }
        result++;
      }
      output.append(std::move(pred));
    }

    int total = std::accumulate(bbox_count.begin(), bbox_count.end(), 0);
    mmdeploy_pose_detector_release_result(detection, total);
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
      .def("__call__", &PyPoseDedector::Apply, py::arg("imgs"),
           py::arg("vboxes") = std::vector<std::vector<Rect>>());
}

class PythonPoseDetectorRegisterer {
 public:
  PythonPoseDetectorRegisterer() {
    gPythonBindings().emplace("pose_detector", register_python_pose_detector);
  }
};

static PythonPoseDetectorRegisterer python_pose_detector_registerer;

}  // namespace mmdeploy
