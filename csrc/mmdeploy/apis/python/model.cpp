// Copyright (c) OpenMMLab. All rights reserved.

#include "common.h"
#include "mmdeploy/common.hpp"

namespace mmdeploy {

static PythonBindingRegisterer register_model{[](py::module& m) {
  py::class_<Model>(m, "Model")
      .def(py::init([](const py::str& path) {
        MMDEPLOY_DEBUG("py::init([](const py::str& path)");
        return Model(path.cast<std::string>().c_str());
      }))
      .def(py::init([](const py::bytes& buffer) {
        MMDEPLOY_DEBUG("py::init([](const py::bytes& buffer)");
        py::buffer_info info(py::buffer(buffer).request());
        return Model(info.ptr, info.size);
      }));
}};

}  // namespace mmdeploy
