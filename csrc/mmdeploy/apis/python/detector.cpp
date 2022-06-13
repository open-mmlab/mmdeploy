// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/apis/c/detector.h"

#include "common.h"

namespace mmdeploy {

class PyDetector {
 public:
  PyDetector(const char *model_path, const char *device_name, int device_id) {
    auto status = mmdeploy_detector_create_by_path(model_path, device_name, device_id, &handle_);
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to create detector");
    }
  }
  py::list Apply(const std::vector<PyImage> &imgs) {
    std::vector<mm_mat_t> mats;
    mats.reserve(imgs.size());
    for (const auto &img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }
    mm_detect_t *detection{};
    int *result_count{};
    auto status =
        mmdeploy_detector_apply(handle_, mats.data(), (int)mats.size(), &detection, &result_count);
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to apply detector, code: " + std::to_string(status));
    }
    auto output = py::list{};
    auto result = detection;
    for (int i = 0; i < mats.size(); ++i) {
      auto bboxes = py::array_t<float>({result_count[i], 5});
      auto labels = py::array_t<int>(result_count[i]);
      auto masks = std::vector<py::array_t<uint8_t>>{};
      masks.reserve(result_count[i]);
      for (int j = 0; j < result_count[i]; ++j, ++result) {
        auto bbox = bboxes.mutable_data(j);
        bbox[0] = result->bbox.left;
        bbox[1] = result->bbox.top;
        bbox[2] = result->bbox.right;
        bbox[3] = result->bbox.bottom;
        bbox[4] = result->score;
        labels.mutable_at(j) = result->label_id;
        if (result->mask) {
          py::array_t<uint8_t> mask({result->mask->height, result->mask->width});
          memcpy(mask.mutable_data(), result->mask->data, mask.nbytes());
          masks.push_back(std::move(mask));
        } else {
          masks.emplace_back();
        }
      }
      output.append(py::make_tuple(std::move(bboxes), std::move(labels), std::move(masks)));
    }
    mmdeploy_detector_release_result(detection, result_count, (int)mats.size());
    return output;
  }
  ~PyDetector() {
    mmdeploy_detector_destroy(handle_);
    handle_ = {};
  }

 private:
  mm_handle_t handle_{};
};

static void register_python_detector(py::module &m) {
  py::class_<PyDetector>(m, "Detector")
      .def(py::init([](const char *model_path, const char *device_name, int device_id) {
        return std::make_unique<PyDetector>(model_path, device_name, device_id);
      }))
      .def("__call__", &PyDetector::Apply);
}

class PythonDetectorRegisterer {
 public:
  PythonDetectorRegisterer() { gPythonBindings().emplace("detector", register_python_detector); }
};

static PythonDetectorRegisterer python_detector_registerer;

}  // namespace mmdeploy
