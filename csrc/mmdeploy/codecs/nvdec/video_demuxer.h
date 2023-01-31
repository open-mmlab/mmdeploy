// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODECS_NVDEC_VIDEO_DEMUXER_H_
#define MMDEPLOY_SRC_CODECS_NVDEC_VIDEO_DEMUXER_H_

#include <nvcuvid.h>

#include <opencv2/videoio.hpp>
#include <string>

#include "mmdeploy/codecs/decoder.h"
#include "mmdeploy/core/status_code.h"

namespace mmdeploy {

namespace nvdec {

class VideoDemuxer {
 public:
  VideoDemuxer(const std::string& path);

  cudaVideoCodec CodecType();

  Result<void> GetNextPacket(unsigned char** data, size_t* size);

  Result<void> GetInfo(VideoInfo& info);

  ~VideoDemuxer();

 private:
  std::shared_ptr<cv::VideoCapture> cap_;
  cv::Mat raw_frame_;
};

}  // namespace nvdec
}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CODECS_NVDEC_VIDEO_DEMUXER_H_
