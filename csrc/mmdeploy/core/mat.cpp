// Copyright (c) OpenMMLab. All rights reserved.

#include "mat.h"

namespace mmdeploy::framework {

Mat::Mat(int h, int w, PixelFormat format, DataType type, Device device, Allocator allocator)
    : format_(format), type_(type), width_(w), height_(h) {
  int bytes_per_pixel = 0;
  switch (format) {
    case PixelFormat::kGRAYSCALE:
      channel_ = 1;
      bytes_per_pixel = 8;
      break;
    case PixelFormat::kNV12:  // fall through
    case PixelFormat::kNV21:
      channel_ = 1;
      bytes_per_pixel = 12;
      assert(w % 2 == 0);
      break;
    case PixelFormat::kBGR:  // fall through
    case PixelFormat::kRGB:
      channel_ = 3;
      bytes_per_pixel = 24;
      break;
    case PixelFormat::kBGRA:
      channel_ = 4;
      bytes_per_pixel = 32;
      break;
    default:
      throw_exception(eNotSupported);
  }

  size_ = height_ * width_ * channel_;
  bytes_ = height_ * width_ * bytes_per_pixel / 8;

  switch (type) {
    case DataType::kFLOAT:
      bytes_ *= sizeof(float);
      break;
    case DataType::kHALF:
      bytes_ *= 2;
      break;
    case DataType::kINT32:
      bytes_ *= sizeof(int32_t);
      break;
    case DataType::kINT8:
      break;
    default:
      throw_exception(eNotSupported);
      break;
  }
  if (device.platform_id() >= 0 && bytes_ > 0) {
    buf_ = Buffer(device, bytes_, std::move(allocator));
  }
}

Mat::Mat(int h, int w, PixelFormat format, DataType type, std::shared_ptr<void> data, Device device)
    : Mat(h, w, format, type, device) {
  buf_ = Buffer(device, bytes_, std::move(data));
}

Mat::Mat(int h, int w, PixelFormat format, DataType type, void* data, Device device)
    : Mat(h, w, format, type, device) {
  buf_ = Buffer(device, bytes_, data);
}

Device Mat::device() const { return buf_.GetDevice(); }
Buffer& Mat::buffer() { return buf_; }
const Buffer& Mat::buffer() const { return buf_; }

}  // namespace mmdeploy::framework
