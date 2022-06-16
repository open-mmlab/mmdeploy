// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_PREPROCESS_TRANSFORM_UTILS_H
#define MMDEPLOY_PREPROCESS_TRANSFORM_UTILS_H

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"

namespace mmdeploy {

inline std::string PixelFormatToString(PixelFormat fmt) {
  switch (fmt) {
    case PixelFormat::kBGR:
      return "BGR";
    case PixelFormat::kRGB:
      return "RGB";
    case PixelFormat::kGRAYSCALE:
      return "GRAYSCALE";
    case PixelFormat::kNV12:
      return "NV12";
    case PixelFormat::kNV21:
      return "NV21";
    case PixelFormat::kBGRA:
      return "BGRA";
  }
  throw_exception(eInvalidArgument);
}

inline std::string DataTypeToString(DataType dt) {
  switch (dt) {
    case DataType::kFLOAT:
      return "Float";
    case DataType::kHALF:
      return "Half";
    case DataType::kINT8:
      return "Int8";
    case DataType::kINT32:
      return "Int32";
    case DataType::kINT64:
      return "Int64";
  }
  throw_exception(eInvalidArgument);
}

inline void AddTransInfo(Value &trans_info, Value &output) {
  if (!trans_info.contains("static") || !trans_info.contains("runtime_args")) {
    return;
  }
  for (auto &&val : trans_info["static"]) {
    output["trans_info"]["static"].push_back(val);
  }
  for (auto &&val : trans_info["runtime_args"]) {
    output["trans_info"]["runtime_args"].push_back(val);
  }
}

inline bool CheckTraceInfoLengthEqual(Value &output) {
  if (output.contains("trans_info")) {
    auto &trans_info = output["trans_info"];
    if (trans_info.contains("static") && trans_info.contains("runtime_args")) {
      return trans_info["static"].size() == trans_info["runtime_args"].size();
    }
  }
}

}  // namespace mmdeploy

#endif  // MMDEPLOY_PREPROCESS_TRANSFORM_UTILS_H
