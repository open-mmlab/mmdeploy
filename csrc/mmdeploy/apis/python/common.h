// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_APIS_PYTHON_COMMON_H_
#define MMDEPLOY_CSRC_APIS_PYTHON_COMMON_H_

#include <stdexcept>

#include "mmdeploy/common.h"
#include "mmdeploy/core/value.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace mmdeploy::python {

using PyImage = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

std::vector<void (*)(py::module&)>& gPythonBindings();

mmdeploy_mat_t GetMat(const PyImage& img);

py::object ToPyObject(const Value& value);

Value FromPyObject(const py::object& obj);

class PythonBindingRegisterer {
 public:
  explicit PythonBindingRegisterer(void (*register_fn)(py::module& m)) {
    gPythonBindings().push_back(register_fn);
  }
};

}  // namespace mmdeploy::python

#endif  // MMDEPLOY_CSRC_APIS_PYTHON_COMMON_H_
