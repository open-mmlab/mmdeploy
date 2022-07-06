// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/pipeline.h"

#include "common.h"
#include "mmdeploy/common.h"

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

  py::object Apply(const py::object& input) {
    auto input_value = FromPyObject(input);
    mmdeploy_value_t output_value{};
    auto status = mmdeploy_pipeline_apply(pipeline_, (mmdeploy_value_t)&input_value, &output_value);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to apply pipeline, code = "s + std::to_string(status));
    }
    return ToPyObject(*(Value*)output_value);
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
      .def("__call__", &PyPipeline::Apply);
}

class PythonPipelineRegisterer {
 public:
  PythonPipelineRegisterer() { gPythonBindings().emplace("pipeline", register_python_pipeline); }
};

static PythonPipelineRegisterer python_pipeline_registerer;

}  // namespace mmdeploy
