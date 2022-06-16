// Copyright (c) OpenMMLab. All rights reserved.
#include <pybind11/pybind11.h>

#include <string>

#include "optimizer.h"
#include "passes/onnx/flatten_cls_head.h"
#include "passes/onnx/merge_shape_concate.h"
#include "passes/onnx/onnx_peephole.h"

namespace mmdeploy {
namespace torch_jit {

void optimize_for_backend(torch::jit::Module& model, const std::string& ir = "torchscript",
                          const std::string& backend = "torchscript") {
  if (ir == "torchscript") {
    model = optimize_for_torchscript(model);
  } else if (ir == "onnx") {
    model = optimize_for_onnx(model);
  } else {
    fprintf(stderr, "No optimize for combination ir: %s backend: %s\n", ir.c_str(),
            backend.c_str());
    exit(-1);
  }
}

PYBIND11_MODULE(ts_optimizer, m) {
  namespace py = pybind11;
  m.def("optimize_for_backend", optimize_for_backend, py::arg("module"),
        py::arg("ir") = std::string("torchscript"),
        py::arg("backend") = std::string("torchscript"));
  py::module_ onnx_module = m.def_submodule("onnx");
  onnx_module.def("_jit_pass_merge_shape_concate", MergeShapeConcate, py::arg("graph"));
  onnx_module.def("_jit_pass_onnx_peephole", ONNXPeephole, py::arg("graph"));
  onnx_module.def("_jit_pass_flatten_cls_head", FlattenClsHead, py::arg("graph"));
}

}  // namespace torch_jit
}  // namespace mmdeploy
