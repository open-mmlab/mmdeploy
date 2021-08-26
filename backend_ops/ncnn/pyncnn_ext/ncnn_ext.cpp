#include <pybind11/pybind11.h>

#include "../ops/ncnn_ops_register.h"
#include "net.h"

PYBIND11_MODULE(ncnn_ext, m) {
  m.def(
      "register_mm_custom_layers",
      [](ncnn::Net &net) { return register_mm_custom_layers(net); },
      "register all mmlab custom ncnn layers.");
}
