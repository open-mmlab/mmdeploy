// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/pipeline.h"

#include "common.h"
#include "mmdeploy/common.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy {

using namespace std::literals;

class PyPipeline {
 public:
  PyPipeline(const py::dict& config, const char* device_name, int device_id) {
    auto value = FromPyObject(config);

    auto status = mmdeploy_pipeline_create((mmdeploy_value_t)&value, device_name, device_id,
                                           nullptr, &pipeline_);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to create pipeline, code = "s + std::to_string(status));
    }
  }

  py::tuple Apply2(const py::tuple& args, bool is_batch) {
    auto inputs = FromPyObject(args);
    if (!is_batch) {
      for (auto& input : inputs) {
        input = Value::Array{std::move(input)};
      }
    }
    mmdeploy_value_t outputs_ptr{};
    auto status = mmdeploy_pipeline_apply(pipeline_, (mmdeploy_value_t)&inputs, &outputs_ptr);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to apply pipeline, code = "s + std::to_string(status));
    }
    auto& outputs = *(Value*)outputs_ptr;
    if (!is_batch) {
      for (auto& output : outputs) {
        output = std::move(output[0]);
      }
    }
    py::tuple rets(outputs.size());
    for (int i = 0; i < outputs.size(); ++i) {
      rets[i] = ToPyObject(outputs[i]);
    }
    mmdeploy_value_destroy(outputs_ptr);
    return rets;
  }

  ~PyPipeline() {
    mmdeploy_pipeline_destroy(pipeline_);
    pipeline_ = {};
  }

 private:
  mmdeploy_pipeline_t pipeline_{};
};

static void register_python_pipeline(py::module& m) {
  py::class_<PyPipeline>(m, "Pipeline")
      .def(py::init([](const py::object& config, const char* device_name, int device_id) {
        return std::make_unique<PyPipeline>(config, device_name, device_id);
      }))
      .def("__call__",
           [](PyPipeline* pipeline, const py::args& args) { return pipeline->Apply2(args, false); })
      .def("batch",
           [](PyPipeline* pipeline, const py::args& args) { return pipeline->Apply2(args, true); });
}

class PythonPipelineRegisterer {
 public:
  PythonPipelineRegisterer() { gPythonBindings().emplace("pipeline", register_python_pipeline); }
};

static PythonPipelineRegisterer python_pipeline_registerer;

}  // namespace mmdeploy
