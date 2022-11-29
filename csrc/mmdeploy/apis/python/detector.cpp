// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/detector.h"

#include "common.h"

namespace mmdeploy::python {

class PyDetector {
 public:
  PyDetector(const char* model_path, const char* device_name, int device_id) {
    auto status = mmdeploy_detector_create_by_path(model_path, device_name, device_id, &detector_);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to create detector");
    }
  }
  py::list Apply(const std::vector<PyImage>& imgs) {
    std::vector<mmdeploy_mat_t> mats;
    mats.reserve(imgs.size());
    for (const auto& img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }
    mmdeploy_detection_t* detection{};
    int* result_count{};
    auto status = mmdeploy_detector_apply(detector_, mats.data(), (int)mats.size(), &detection,
                                          &result_count);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to apply detector, code: " + std::to_string(status));
    }
    using Sptr = std::shared_ptr<mmdeploy_detection_t>;
    Sptr holder(detection, [result_count, n = mats.size()](auto p) {
      mmdeploy_detector_release_result(p, result_count, n);
    });
    auto output = py::list{};
    auto result = detection;
    for (int i = 0; i < mats.size(); ++i) {
      auto bboxes = py::array_t<float>({result_count[i], 5});
      auto labels = py::array_t<int>(result_count[i]);
      auto masks = std::vector<py::array>();
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
          masks.emplace_back(std::array{result->mask->height, result->mask->width},  // shape
                             reinterpret_cast<uint8_t*>(result->mask->data),         // data
                             py::capsule(new Sptr(holder),                           // handle
                                         [](void* p) { delete reinterpret_cast<Sptr*>(p); }));
        } else {
          masks.emplace_back();
        }
      }
      output.append(py::make_tuple(std::move(bboxes), std::move(labels), std::move(masks)));
    }
    return output;
  }
  ~PyDetector() {
    mmdeploy_detector_destroy(detector_);
    detector_ = {};
  }

 private:
  mmdeploy_detector_t detector_{};
};

static PythonBindingRegisterer register_detector{[](py::module& m) {
  py::class_<PyDetector>(m, "Detector")
      .def(py::init([](const char* model_path, const char* device_name, int device_id) {
             return std::make_unique<PyDetector>(model_path, device_name, device_id);
           }),
           py::arg("model_path"), py::arg("device_name"), py::arg("device_id") = 0)
      .def("__call__",
           [](PyDetector* self, const PyImage& img) -> py::tuple {
             return self->Apply(std::vector{img})[0];
           })
      .def("batch", &PyDetector::Apply);
}};

}  // namespace mmdeploy::python
