// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/pipeline.hpp"

#include "common.h"
#include "mmdeploy/common.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy::python {

using namespace std::literals;

static PythonBindingRegisterer register_pipeline{[](py::module& m) {
  py::class_<Pipeline>(m, "Pipeline")
      .def(py::init([](const py::object& config, const Context& context) {
        auto _config = FromPyObject(config);
        return std::make_unique<Pipeline>(_config, context);
      }))
      .def("__call__",
           [](Pipeline* pipeline, const py::args& args) {
             auto inputs = FromPyObject(args);
             for (auto& input : inputs) {
               input = Value::Array{std::move(input)};
             }
             auto outputs = pipeline->Apply(inputs);
             for (auto& output : outputs) {
               output = std::move(output[0]);
             }
             py::tuple rets(outputs.size());
             for (int i = 0; i < outputs.size(); ++i) {
               rets[i] = ToPyObject(outputs[i]);
             }
             return rets;
           })
      .def("batch", [](Pipeline* pipeline, const py::args& args) {
        auto inputs = FromPyObject(args);
        auto outputs = pipeline->Apply(inputs);
        py::tuple rets(outputs.size());
        for (int i = 0; i < outputs.size(); ++i) {
          rets[i] = ToPyObject(outputs[i]);
        }
        return rets;
      });
}};

}  // namespace mmdeploy::python
