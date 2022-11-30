// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codecs/nvdec/video_demuxer.h"

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/status_code.h"

#ifndef CV_FOURCC_MACRO
#define CV_FOURCC_MACRO(c1, c2, c3, c4) \
  (((c1)&255) + (((c2)&255) << 8) + (((c3)&255) << 16) + (((c4)&255) << 24))
#endif

namespace mmdeploy {

namespace nvdec {

VideoDemuxer::VideoDemuxer(const std::string& path) {
  cap_.reset(new cv::VideoCapture(path));
  if (!cap_->isOpened()) {
    MMDEPLOY_ERROR("Can't open {}", path);
    throw_exception(eFail);
  }
  if (!cap_->set(cv::CAP_PROP_FORMAT, -1)) {
    MMDEPLOY_ERROR("Can't set raw mode");
    throw_exception(eFail);
  };
}

cudaVideoCodec VideoDemuxer::CodecType() {
  int codec = (int)cap_->get(cv::CAP_PROP_FOURCC);

  switch (codec) {
    case CV_FOURCC_MACRO('m', 'p', 'e', 'g'):
    case CV_FOURCC_MACRO('m', 'p', 'g', '1'):
    case CV_FOURCC_MACRO('M', 'P', 'G', '1'):
      return cudaVideoCodec_MPEG1;
    case CV_FOURCC_MACRO('m', 'p', 'g', '2'):
    case CV_FOURCC_MACRO('M', 'P', 'G', '2'):
      return cudaVideoCodec_MPEG2;
    case CV_FOURCC_MACRO('X', 'V', 'I', 'D'):
    case CV_FOURCC_MACRO('m', 'p', '4', 'v'):
    case CV_FOURCC_MACRO('D', 'I', 'V', 'X'):
      return cudaVideoCodec_MPEG4;
    case CV_FOURCC_MACRO('W', 'V', 'C', '1'):
      return cudaVideoCodec_VC1;
    case CV_FOURCC_MACRO('H', '2', '6', '4'):
    case CV_FOURCC_MACRO('h', '2', '6', '4'):
    case CV_FOURCC_MACRO('a', 'v', 'c', '1'):
      return cudaVideoCodec_H264;
    case CV_FOURCC_MACRO('H', '2', '6', '5'):
    case CV_FOURCC_MACRO('h', '2', '6', '5'):
    case CV_FOURCC_MACRO('h', 'e', 'v', 'c'):
      return cudaVideoCodec_HEVC;
    case CV_FOURCC_MACRO('M', 'J', 'P', 'G'):
      return cudaVideoCodec_JPEG;
    case CV_FOURCC_MACRO('v', 'p', '8', '0'):
    case CV_FOURCC_MACRO('V', 'P', '8', '0'):
    case CV_FOURCC_MACRO('v', 'p', '0', '8'):
    case CV_FOURCC_MACRO('V', 'P', '0', '8'):
      return cudaVideoCodec_VP8;
    case CV_FOURCC_MACRO('v', 'p', '9', '0'):
    case CV_FOURCC_MACRO('V', 'P', '9', '0'):
    case CV_FOURCC_MACRO('V', 'P', '0', '9'):
    case CV_FOURCC_MACRO('v', 'p', '0', '9'):
      return cudaVideoCodec_VP9;
    case CV_FOURCC_MACRO('a', 'v', '1', '0'):
    case CV_FOURCC_MACRO('A', 'V', '1', '0'):
    case CV_FOURCC_MACRO('a', 'v', '0', '1'):
    case CV_FOURCC_MACRO('A', 'V', '0', '1'):
      return cudaVideoCodec_AV1;
    default:
      break;
  }
  MMDEPLOY_ERROR("Codec {} not supported", codec);
  throw_exception(eNotSupported);
}

Result<void> VideoDemuxer::GetNextPacket(unsigned char** data, size_t* size) {
  (*cap_) >> raw_frame_;
  *data = raw_frame_.data;
  *size = raw_frame_.total();
  if (*size != 0) {
    return success();
  }
  return Status(eFail);
}

Result<void> VideoDemuxer::GetInfo(VideoInfo& info) {
  info.width = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
  info.height = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
  info.fourcc = (int)cap_->get(cv::CAP_PROP_FOURCC);
  info.fps = cap_->get(cv::CAP_PROP_FPS);
  return success();
}

VideoDemuxer::~VideoDemuxer() {
  if (cap_->isOpened()) {
    cap_->release();
  }
}

}  // namespace nvdec
}  // namespace mmdeploy
