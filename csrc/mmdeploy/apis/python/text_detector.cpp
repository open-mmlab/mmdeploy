// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/apis/c/text_detector.h"

#include "common.h"

namespace mmdeploy {

class PyTextDetector {
 public:
  PyTextDetector(const char *model_path, const char *device_name, int device_id) {
    auto status =
        mmdeploy_text_detector_create_by_path(model_path, device_name, device_id, &handle_);
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to create text_detector");
    }
  }
  std::vector<py::array_t<float>> Apply(const std::vector<PyImage> &imgs) {
    std::vector<mm_mat_t> mats;
    mats.reserve(imgs.size());
    for (const auto &img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }
    mm_text_detect_t *detection{};
    int *result_count{};
    auto status = mmdeploy_text_detector_apply(handle_, mats.data(), (int)mats.size(), &detection,
                                               &result_count);
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to apply text_detector, code: " + std::to_string(status));
    }
    auto output = std::vector<py::array_t<float>>{};
    auto result = detection;
    for (int i = 0; i < mats.size(); ++i) {
      auto bboxes = py::array_t<float>({result_count[i], 9});
      for (int j = 0; j < result_count[i]; ++j, ++result) {
        auto data = bboxes.mutable_data(j);
        for (const auto &p : result->bbox) {
          *data++ = p.x;
          *data++ = p.y;
        }
        *data++ = result->score;
      }
      output.push_back(std::move(bboxes));
    }
    mmdeploy_text_detector_release_result(detection, result_count, (int)mats.size());
    return output;
  }
  ~PyTextDetector() {
    mmdeploy_text_detector_destroy(handle_);
    handle_ = {};
  }

 private:
  mm_handle_t handle_{};
};

static void register_python_text_detector(py::module &m) {
  py::class_<PyTextDetector>(m, "TextDetector")
      .def(py::init([](const char *model_path, const char *device_name, int device_id) {
        return std::make_unique<PyTextDetector>(model_path, device_name, device_id);
      }))
      .def("__call__", &PyTextDetector::Apply);
}

class PythonTextDetectorRegisterer {
 public:
  PythonTextDetectorRegisterer() {
    gPythonBindings().emplace("text_detector", register_python_text_detector);
  }
};

static PythonTextDetectorRegisterer python_text_detector_registerer;

}  // namespace mmdeploy
