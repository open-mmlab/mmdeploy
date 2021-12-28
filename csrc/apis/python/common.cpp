// Copyright (c) OpenMMLab. All rights reserved.

#include "apis/python/common.h"

namespace mmdeploy {

std::map<std::string, void (*)(py::module &)> &gPythonBindings() {
  static std::map<std::string, void (*)(py::module &)> v;
  return v;
}

mm_mat_t GetMat(const PyImage &img) {
  auto info = img.request();
  if (info.ndim != 3) {
    fprintf(stderr, "info.ndim = %d\n", (int)info.ndim);
    throw std::runtime_error("continuous uint8 HWC array expected");
  }
  auto channels = (int)info.shape[2];
  mm_mat_t mat{};
  if (channels == 1) {
    mat.format = MM_GRAYSCALE;
  } else if (channels == 3) {
    mat.format = MM_BGR;
  } else {
    throw std::runtime_error("images of 1 or 3 channels are supported");
  }
  mat.height = (int)info.shape[0];
  mat.width = (int)info.shape[1];
  mat.channel = channels;
  mat.type = MM_INT8;
  mat.data = (uint8_t *)info.ptr;
  return mat;
}

}  // namespace mmdeploy

PYBIND11_MODULE(mmdeploy_python, m) {
  for (const auto &[_, f] : mmdeploy::gPythonBindings()) {
    f(m);
  }
}
