// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/model.h"

#include "common.h"

namespace mmdeploy {

static void register_python_model(py::module& m) {
  using mmdeploy::Model;
  py::class_<Model>(m, "Model")
      .def(py::init([](const py::str& path) {
        MMDEPLOY_ERROR("py::init([](const py::str& path)");
        return Model(path.cast<std::string>());
      }))
      .def(py::init([](const py::bytes& buffer) {
        MMDEPLOY_ERROR("py::init([](const py::bytes& buffer)");
        py::buffer_info info(py::buffer(buffer).request());
        return Model(info.ptr, info.size);
      }));
}


class PythonModelRegisterer {
 public:
  PythonModelRegisterer() { gPythonBindings().emplace("model", register_python_model); }
};

static PythonModelRegisterer python_model_registerer;

}