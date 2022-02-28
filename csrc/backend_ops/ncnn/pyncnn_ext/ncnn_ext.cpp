// Copyright (c) OpenMMLab. All rights reserved.
#include <pybind11/pybind11.h>

#include "ncnn_ops_register.h"
#include "net.h"

PYBIND11_MODULE(ncnn_ext, m) {
  m.def(
      "register_mmdeploy_custom_layers",
      [](ncnn::Net &net) { return register_mmdeploy_custom_layers(net); },
      "register mmdeploy custom ncnn layers.");
}
