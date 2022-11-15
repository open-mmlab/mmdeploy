// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODECS_NVJPEG_DECODE_JPEG_H_
#define MMDEPLOY_SRC_CODECS_NVJPEG_DECODE_JPEG_H_

#include <memory>

#include "mmdeploy/core/macro.h"
#include "mmdeploy/core/status_code.h"
#include "mmdeploy/core/types.h"
#include "mmdeploy/core/value.h"

namespace mmdeploy {
namespace codecs {

class MMDEPLOY_API JPEGDecoder {
 public:
  JPEGDecoder(int device_id = 0);

  ~JPEGDecoder();

  Result<Value> Apply(const std::vector<const char*>& raw_data, const std::vector<int>& length,
                      PixelFormat format);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace codecs
}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CODECS_NVJPEG_DECODE_JPEG_H_