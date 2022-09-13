// Copyright (c) OpenMMLab. All rights reserved.

#include <optional>

#include "common.h"
#include "mmdeploy/common.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/value.h"

namespace mmdeploy {

namespace python {

framework::Mat _get_mat(const PyImage& img) {
  auto info = img.request();
  if (info.ndim != 3) {
    fprintf(stderr, "info.ndim = %d\n", (int)info.ndim);
    throw std::runtime_error("continuous uint8 HWC array expected");
  }
  auto channels = (int)info.shape[2];
  PixelFormat format;
  if (channels == 1) {
    format = PixelFormat::kGRAYSCALE;
  } else if (channels == 3) {
    format = PixelFormat::kBGR;
  } else {
    throw std::runtime_error("images of 1 or 3 channels are supported");
  }

  return {
      (int)info.shape[0],                             // height
      (int)info.shape[1],                             // width
      format,                                         // format
      DataType::kINT8,                                // type
      std::shared_ptr<void>(info.ptr, [](void*) {}),  // data
      framework::Device(0),                           // device
  };
}

std::optional<Value> _to_value_internal(const void* object, mmdeploy_context_type_t type) {
  switch (type) {
    case MMDEPLOY_TYPE_MODEL:
      return Value(*(const framework::Model*)object);
    case MMDEPLOY_TYPE_DEVICE:
      return Value(*(const framework::Device*)object);
    case MMDEPLOY_TYPE_MAT:
      return _get_mat(*(const py::array*)object);
    default:
      return std::nullopt;
  }
}

}  // namespace python

}  // namespace mmdeploy
