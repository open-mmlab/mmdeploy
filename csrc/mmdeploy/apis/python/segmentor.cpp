// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/segmentor.h"

#include "common.h"

namespace mmdeploy::python {

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

  std::vector<py::array> Apply(const std::vector<PyImage>& imgs) {
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
    using Sptr = std::shared_ptr<mmdeploy_segmentation_t>;
    Sptr holder(segm, [n = mats.size()](auto p) { mmdeploy_segmentor_release_result(p, n); });

    std::vector<py::array> rets(mats.size());
    for (size_t i = 0; i < mats.size(); ++i) {
      rets[i] = {
          {segm[i].height, segm[i].width},                                 // shape
          segm[i].mask,                                                    // data
          py::capsule(new Sptr(holder),                                    // handle
                      [](void* p) { delete reinterpret_cast<Sptr*>(p); })  //
      };
    }
    return rets;
  }

 private:
  mmdeploy_segmentor_t segmentor_{};
};

static PythonBindingRegisterer register_segmentor{[](py::module& m) {
  py::class_<PySegmentor>(m, "Segmentor")
      .def(py::init([](const char* model_path, const char* device_name, int device_id) {
             return std::make_unique<PySegmentor>(model_path, device_name, device_id);
           }),
           py::arg("model_path"), py::arg("device_name"), py::arg("device_id") = 0)
      .def("__call__",
           [](PySegmentor* self, const PyImage& img) -> py::array {
             return self->Apply(std::vector{img})[0];
           })
      .def("batch", &PySegmentor::Apply);
}};

}  // namespace mmdeploy::python
