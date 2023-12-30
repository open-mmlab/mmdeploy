// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_MMDEPLOY_UTILS_H
#define MMDEPLOY_MMDEPLOY_UTILS_H

#include "mmdeploy/core/types.h"

namespace triton::backend::mmdeploy {

inline TRITONSERVER_DataType ConvertDataType(::mmdeploy::DataType data_type) {
  using namespace ::mmdeploy::data_types;
  switch (data_type) {
    case kFLOAT:
      return TRITONSERVER_TYPE_FP32;
    case kHALF:
      return TRITONSERVER_TYPE_FP16;
    case kINT8:
      return TRITONSERVER_TYPE_UINT8;
    case kINT32:
      return TRITONSERVER_TYPE_INT32;
    case kINT64:
      return TRITONSERVER_TYPE_INT64;
    default:
      return TRITONSERVER_TYPE_INVALID;
  }
}

inline ::mmdeploy::DataType ConvertDataType(TRITONSERVER_DataType data_type) {
  using namespace ::mmdeploy::data_types;
  switch (data_type) {
    case TRITONSERVER_TYPE_FP32:
      return kFLOAT;
    case TRITONSERVER_TYPE_FP16:
      return kHALF;
    case TRITONSERVER_TYPE_UINT8:
      return kINT8;
    case TRITONSERVER_TYPE_INT32:
      return kINT32;
    case TRITONSERVER_TYPE_INT64:
      return kINT64;
    default:
      return ::mmdeploy::DataType::kCOUNT;
  }
}

}  // namespace triton::backend::mmdeploy

#endif  // MMDEPLOY_MMDEPLOY_UTILS_H
