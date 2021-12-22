// Copyright (c) OpenMMLab. All rights reserved.

#include <stdexcept>

#include "detector.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

using PyImage = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

namespace mmdeploy {

mm_mat_t GetMat(const PyImage &img) {
  auto info = img.request();
  if (info.ndim != 3) {
    fprintf(stderr, "info.ndim = %d\n", (int)info.ndim);
    throw std::runtime_error("continuous uint8 HWC array expected");
  }
  auto channels = (int)info.shape[2];
  mm_mat_t mat{};
  if (channels == 1) {
    mat.format = MM_GRAYSCALE;
  } else if (channels == 3) {
    mat.format = MM_BGR;
  } else {
    throw std::runtime_error("images of 1 or 3 channels are supported");
  }
  mat.height = (int)info.shape[0];
  mat.width = (int)info.shape[1];
  mat.channel = channels;
  mat.type = MM_INT8;
  mat.data = (uint8_t *)info.ptr;
  return mat;
}

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
      for (int j = 0; j < result_count[i]; ++j, ++result) {
        auto bbox = bboxes.mutable_data(j);
        bbox[0] = result->bbox.left;
        bbox[1] = result->bbox.top;
        bbox[2] = result->bbox.right;
        bbox[3] = result->bbox.bottom;
        bbox[4] = result->score;
        labels.mutable_at(j) = result->label_id;
      }
      output.append(py::make_tuple(std::move(bboxes), std::move(labels)));
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

}  // namespace mmdeploy

using mmdeploy::PyDetector;

PYBIND11_MODULE(mmdeploy_python, m) {
  py::class_<PyDetector>(m, "Detector")
      .def(py::init([](const char *model_path, const char *device_name, int device_id) {
        return std::make_unique<PyDetector>(model_path, device_name, device_id);
      }))
      .def("__call__", &PyDetector::Apply);
}
