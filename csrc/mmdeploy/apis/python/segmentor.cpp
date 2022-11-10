// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/segmentor.h"

#include "common.h"

namespace mmdeploy::python {

class Segmentation {
 public:
  explicit Segmentation(std::shared_ptr<mmdeploy_segmentation_t> ptr, int index)
      : ptr_(std::move(ptr)), index_(index) {}

  mmdeploy_segmentation_t* operator->() { return ptr_.get() + index_; }

 private:
  std::shared_ptr<mmdeploy_segmentation_t> ptr_;
  int index_;
};

class PySegmentor {
 public:
  PySegmentor(const char* model_path, const char* device_name, int device_id) {
    auto status =
        mmdeploy_segmentor_create_by_path(model_path, device_name, device_id, &segmentor_);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to create segmentor");
    }
  }
  ~PySegmentor() {
    mmdeploy_segmentor_destroy(segmentor_);
    segmentor_ = {};
  }

  std::vector<Segmentation> Apply(const std::vector<PyImage>& imgs) {
    std::vector<mmdeploy_mat_t> mats;
    mats.reserve(imgs.size());
    for (const auto& img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }
    mmdeploy_segmentation_t* segm{};
    auto status = mmdeploy_segmentor_apply(segmentor_, mats.data(), (int)mats.size(), &segm);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to apply segmentor, code: " + std::to_string(status));
    }
    std::shared_ptr<mmdeploy_segmentation_t> result(
        segm, [n = mats.size()](auto p) { mmdeploy_segmentor_release_result(p, n); });
    std::vector<Segmentation> rets;
    rets.reserve(mats.size());
    for (size_t i = 0; i < mats.size(); ++i) {
      rets.emplace_back(result, i);
    }
    return rets;
  }

 private:
  mmdeploy_segmentor_t segmentor_{};
};

static PythonBindingRegisterer register_segmentor{[](py::module& m) {
  py::class_<Segmentation>(m, "Segmentation", py::buffer_protocol())
      .def_buffer([](Segmentation& s) -> py::buffer_info {
        return py::buffer_info(s->mask,                               // data pointer
                               sizeof(int),                           // size of scalar
                               py::format_descriptor<int>::format(),  // format desc
                               2,                                     // num of dims
                               {s->height, s->width},                 // shape
                               {sizeof(int) * s->width, sizeof(int)}  // stride
        );
      });

  py::class_<PySegmentor>(m, "Segmentor")
      .def(py::init([](const char* model_path, const char* device_name, int device_id) {
             return std::make_unique<PySegmentor>(model_path, device_name, device_id);
           }),
           py::arg("model_path"), py::arg("device_name"), py::arg("device_id") = 0)
      .def("__call__",
           [](PySegmentor* self, const PyImage& img) -> Segmentation {
             return self->Apply(std::vector{img})[0];
           })
      .def("batch", &PySegmentor::Apply);
}};

}  // namespace mmdeploy::python
