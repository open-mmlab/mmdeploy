// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODECS_NVJPEG_DECODER_H_
#define MMDEPLOY_SRC_CODECS_NVJPEG_DECODER_H_

#include <memory>

#include "mmdeploy/codecs/decoder.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/macro.h"
#include "mmdeploy/core/status_code.h"
#include "mmdeploy/core/types.h"
#include "mmdeploy/core/value.h"

namespace mmdeploy {
namespace nvjpeg {

class ImageDecoder : public ::mmdeploy::ImageDecoder {
 public:
  ImageDecoder();

  ~ImageDecoder();

  Result<void> Init(const Value& cfg);

  Result<Value> Process(const Value& input);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nvjpeg
}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CODECS_NVJPEG_DECODER_H_
