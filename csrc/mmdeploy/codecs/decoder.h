// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODECS_DECODER_H_
#define MMDEPLOY_SRC_CODECS_DECODER_H_

#include <memory>

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/macro.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/types.h"
#include "mmdeploy/core/value.h"

namespace mmdeploy {

using Decoder = Module;
using DecoderCreator = Creator<Decoder>;

struct ImageDecoderInput {
  const char* raw_data;
  int length;
  PixelFormat format;
};

struct VideoInfo {
  int width;
  int height;
  int fourcc;
  double fps;
};

class MMDEPLOY_API ImageDecoder : public Decoder {
 public:
  virtual Result<void> Init(const Value& cfg) = 0;
};

class MMDEPLOY_API VideoDecoder {
 public:
  virtual Result<void> Init(const Value& cfg) = 0;

  virtual Result<void> GetInfo(VideoInfo& info) = 0;

  virtual Result<void> Read(framework::Mat& out) = 0;

  virtual Result<void> Retrieve(framework::Mat& out) = 0;

  virtual Result<void> Grab() = 0;
};

MMDEPLOY_DECLARE_REGISTRY(ImageDecoder, std::unique_ptr<ImageDecoder>(const Value& config));
MMDEPLOY_DECLARE_REGISTRY(VideoDecoder, std::unique_ptr<VideoDecoder>(const Value& config));
MMDEPLOY_REGISTER_TYPE_ID(ImageDecoderInput, 11);

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CODECS_DECODER_H_
