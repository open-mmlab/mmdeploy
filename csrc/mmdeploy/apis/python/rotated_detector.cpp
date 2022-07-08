// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/apis/c/rotated_detector.h"

#include "common.h"

namespace mmdeploy {

class PyRotatedDetector {
 public:
  PyRotatedDetector(const char *model_path, const char *device_name, int device_id) {
    auto status =
        mmdeploy_rotated_detector_create_by_path(model_path, device_name, device_id, &handle_);
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to create rotated detector");
    }
  }
  py::list Apply(const std::vector<PyImage> &imgs) {
    std::vector<mm_mat_t> mats;
    mats.reserve(imgs.size());
    for (const auto &img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }

    mm_rotated_detect_t *rbboxes{};
    int *res_count{};
    auto status = mmdeploy_rotated_detector_apply(handle_, mats.data(), (int)mats.size(), &rbboxes,
                                                  &res_count);
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to apply rotated detector, code: " + std::to_string(status));
    }
    auto output = py::list{};
    auto result = rbboxes;
    auto counts = res_count;
    for (int i = 0; i < mats.size(); i++) {
      auto _dets = py::array_t<float>({*counts, 6});
      auto _labels = py::array_t<int>({*counts});
      auto dets = _dets.mutable_data();
      auto labels = _labels.mutable_data();
      for (int j = 0; j < *counts; j++) {
        for (int k = 0; k < 5; k++) {
          *dets++ = result->rbbox[k];
        }
        *dets++ = result->score;
        *labels++ = result->label_id;
        result++;
      }
      counts++;
      output.append(py::make_tuple(std::move(_dets), std::move(_labels)));
    }
    mmdeploy_rotated_detector_release_result(rbboxes, res_count);
    return output;
  }
  ~PyRotatedDetector() {
    mmdeploy_rotated_detector_destroy(handle_);
    handle_ = {};
  }

 private:
  mm_handle_t handle_{};
};

static void register_python_rotated_detector(py::module &m) {
  py::class_<PyRotatedDetector>(m, "RotatedDetector")
      .def(py::init([](const char *model_path, const char *device_name, int device_id) {
        return std::make_unique<PyRotatedDetector>(model_path, device_name, device_id);
      }))
      .def("__call__", &PyRotatedDetector::Apply);
}

class PythonRotatedDetectorRegisterer {
 public:
  PythonRotatedDetectorRegisterer() {
    gPythonBindings().emplace("rotated_detector", register_python_rotated_detector);
  }
};

static PythonRotatedDetectorRegisterer python_rotated_detector_registerer;

}  // namespace mmdeploy
