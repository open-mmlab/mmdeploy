// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODECS_NVDEC_NVDECODER_H_
#define MMDEPLOY_SRC_CODECS_NVDEC_NVDECODER_H_

#include <nvcuvid.h>

#include <memory>

#include "mmdeploy/codecs/decoder.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/mat.h"

namespace mmdeploy {

namespace nvdec {

class VideoDecoder : public ::mmdeploy::VideoDecoder {
 public:
  VideoDecoder();

  ~VideoDecoder();

  Result<void> Init(const Value& args) override;

  Result<void> GetInfo(VideoInfo& info) override;

  Result<void> Read(framework::Mat& out) override;

  Result<void> Retrieve(framework::Mat& out) override;

  Result<void> Grab() override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nvdec
}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CODECS_NVDEC_NVDECODER_H_
