// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/rotated_detector.h"

#include "common.h"

namespace mmdeploy::python {

class PyRotatedDetector {
 public:
  PyRotatedDetector(const char* model_path, const char* device_name, int device_id) {
    auto status =
        mmdeploy_rotated_detector_create_by_path(model_path, device_name, device_id, &detector_);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to create rotated detector");
    }
  }
  py::list Apply(const std::vector<PyImage>& imgs) {
    std::vector<mmdeploy_mat_t> mats;
    mats.reserve(imgs.size());
    for (const auto& img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }

    mmdeploy_rotated_detection_t* rbboxes{};
    int* res_count{};
    auto status = mmdeploy_rotated_detector_apply(detector_, mats.data(), (int)mats.size(),
                                                  &rbboxes, &res_count);
    if (status != MMDEPLOY_SUCCESS) {
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
    mmdeploy_rotated_detector_destroy(detector_);
    detector_ = {};
  }

 private:
  mmdeploy_rotated_detector_t detector_{};
};

static PythonBindingRegisterer register_rotated_detector{[](py::module& m) {
  py::class_<PyRotatedDetector>(m, "RotatedDetector")
      .def(py::init([](const char* model_path, const char* device_name, int device_id) {
             return std::make_unique<PyRotatedDetector>(model_path, device_name, device_id);
           }),
           py::arg("model_path"), py::arg("device_name"), py::arg("device_id") = 0)
      .def("__call__",
           [](PyRotatedDetector* self, const PyImage& img) -> py::tuple {
             return self->Apply(std::vector{img})[0];
           })
      .def("batch", &PyRotatedDetector::Apply);
}};

}  // namespace mmdeploy::python
