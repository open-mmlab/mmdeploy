// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/apis/c/restorer.h"

#include "common.h"

namespace mmdeploy {

class PyRestorer {
 public:
  PyRestorer(const char *model_path, const char *device_name, int device_id) {
    auto status = mmdeploy_restorer_create_by_path(model_path, device_name, device_id, &handle_);
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to create restorer");
    }
  }
  ~PyRestorer() {
    mmdeploy_restorer_destroy(handle_);
    handle_ = {};
  }

  std::vector<py::array_t<uint8_t>> Apply(const std::vector<PyImage> &imgs) {
    std::vector<mm_mat_t> mats;
    mats.reserve(imgs.size());
    for (const auto &img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }
    mm_mat_t *results{};
    auto status = mmdeploy_restorer_apply(handle_, mats.data(), (int)mats.size(), &results);
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to apply restorer, code: " + std::to_string(status));
    }
    auto output = std::vector<py::array_t<uint8_t>>{};
    output.reserve(mats.size());
    for (int i = 0; i < mats.size(); ++i) {
      py::array_t<uint8_t> restored({results[i].height, results[i].width, results[i].channel});
      memcpy(restored.mutable_data(), results[i].data, restored.nbytes());
      output.push_back(std::move(restored));
    }
    mmdeploy_restorer_release_result(results, (int)mats.size());
    return output;
  }

 private:
  mm_handle_t handle_{};
};

static void register_python_restorer(py::module &m) {
  py::class_<PyRestorer>(m, "Restorer")
      .def(py::init([](const char *model_path, const char *device_name, int device_id) {
        return std::make_unique<PyRestorer>(model_path, device_name, device_id);
      }))
      .def("__call__", &PyRestorer::Apply);
}

class PythonRestorerRegisterer {
 public:
  PythonRestorerRegisterer() { gPythonBindings().emplace("restorer", register_python_restorer); }
};

static PythonRestorerRegisterer python_restorer_registerer;

}  // namespace mmdeploy
