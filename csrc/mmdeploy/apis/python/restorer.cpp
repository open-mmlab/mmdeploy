// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/restorer.h"

#include "common.h"

namespace mmdeploy::python {

class PyRestorer {
 public:
  PyRestorer(const char* model_path, const char* device_name, int device_id) {
    auto status = mmdeploy_restorer_create_by_path(model_path, device_name, device_id, &restorer_);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to create restorer");
    }
  }
  ~PyRestorer() {
    mmdeploy_restorer_destroy(restorer_);
    restorer_ = {};
  }

  std::vector<py::array> Apply(const std::vector<PyImage>& imgs) {
    std::vector<mmdeploy_mat_t> mats;
    mats.reserve(imgs.size());
    for (const auto& img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }
    mmdeploy_mat_t* results{};
    auto status = mmdeploy_restorer_apply(restorer_, mats.data(), (int)mats.size(), &results);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to apply restorer, code: " + std::to_string(status));
    }
    using Sptr = std::shared_ptr<mmdeploy_mat_t>;
    Sptr holder(results, [n = mats.size()](auto p) { mmdeploy_restorer_release_result(p, n); });

    std::vector<py::array> rets(mats.size());
    for (int i = 0; i < mats.size(); ++i) {
      rets[i] = {
          {results[i].height, results[i].width, results[i].channel},       // shape
          results[i].data,                                                 // data
          py::capsule(new Sptr(holder),                                    // handle
                      [](void* p) { delete reinterpret_cast<Sptr*>(p); })  //
      };
    }
    return rets;
  }

 private:
  mmdeploy_restorer_t restorer_{};
};

static PythonBindingRegisterer register_restorer{[](py::module& m) {
  py::class_<PyRestorer>(m, "Restorer")
      .def(py::init([](const char* model_path, const char* device_name, int device_id) {
             return std::make_unique<PyRestorer>(model_path, device_name, device_id);
           }),
           py::arg("model_path"), py::arg("device_name"), py::arg("device_id") = 0)
      .def("__call__",
           [](PyRestorer* self, const PyImage& img) -> py::array {
             return self->Apply(std::vector{img})[0];
           })
      .def("batch", &PyRestorer::Apply);
}};

}  // namespace mmdeploy::python
