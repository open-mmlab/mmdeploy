// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/vision.h"

namespace mmdeploy::operation::dummy {

namespace {

const Buffer& g_dummy_buffer() {
  static Buffer buffer{Device(0), 0, nullptr};
  return buffer;
}

}  // namespace

class HWC2CHWImpl : public HWC2CHW {
 public:
  Result<void> apply(const Tensor& img, Tensor& dst) override {
    auto& shape = img.shape();
    dst = {{Device{0}, img.data_type(), {shape[0], shape[3], shape[1], shape[2]}},
           g_dummy_buffer()};
    return success();
  }
};
MMDEPLOY_REGISTER_FACTORY_FUNC(HWC2CHW, (dummy, 0),
                               []() { return std::make_unique<HWC2CHWImpl>(); });

class CropImpl : public Crop {
 public:
  Result<void> apply(const Tensor& src, Tensor& dst, int top, int left, int bottom,
                     int right) override {
    auto shape = src.shape();
    shape[1] = bottom - top + 1;  // h
    shape[2] = right - left + 1;  // w
    dst = {{Device{0}, src.data_type(), shape}, g_dummy_buffer()};
    return success();
  }
};
MMDEPLOY_REGISTER_FACTORY_FUNC(Crop, (dummy, 0), []() { return std::make_unique<CropImpl>(); });

class ToFloatImpl : public ToFloat {
 public:
  Result<void> apply(const Tensor& src, Tensor& dst) override {
    dst = {{Device{0}, DataType::kFLOAT, src.shape()}, g_dummy_buffer()};
    return success();
  }
};
MMDEPLOY_REGISTER_FACTORY_FUNC(ToFloat, (dummy, 0),
                               []() { return std::make_unique<ToFloatImpl>(); });

class CvtColorImpl : public CvtColor {
 public:
  Result<void> apply(const Mat& src, Mat& dst, PixelFormat dst_fmt) override {
    dst = {src.height(), src.width(), dst_fmt, src.type(), nullptr, Device{0}};
    return success();
  }
};
MMDEPLOY_REGISTER_FACTORY_FUNC(CvtColor, (dummy, 0),
                               [] { return std::make_unique<CvtColorImpl>(); });

class NormalizeImpl : public Normalize {
 public:
  Result<void> apply(const Tensor& src, Tensor& dst) override {
    dst = {{Device{0}, DataType::kFLOAT, src.shape()}, g_dummy_buffer()};
    return success();
  }
};
MMDEPLOY_REGISTER_FACTORY_FUNC(Normalize, (dummy, 0), [](const Normalize::Param& param) {
  return std::make_unique<NormalizeImpl>();
});

class PadImpl : public Pad {
 public:
  Result<void> apply(const Tensor& src, Tensor& dst, int top, int left, int bottom,
                     int right) override {
    auto shape = src.shape();  // 1HWC
    shape[1] += top + bottom;
    shape[2] += left + right;
    dst = {{Device{0}, src.data_type(), shape}, g_dummy_buffer()};
    return success();
  }
};
MMDEPLOY_REGISTER_FACTORY_FUNC(Pad, (dummy, 0), [](const string_view& border_type, float pad_val) {
  return std::make_unique<PadImpl>();
});

class ResizeImpl : public Resize {
 public:
  Result<void> apply(const Tensor& src, Tensor& dst, int dst_h, int dst_w) override {
    dst = {{Device{0}, dst.data_type(), {1, dst_h, dst_w, src.shape(3)}}, g_dummy_buffer()};
    return success();
  }
};
MMDEPLOY_REGISTER_FACTORY_FUNC(Resize, (dummy, 0), [](const string_view& interp) {
  return std::make_unique<ResizeImpl>();
});

}  // namespace mmdeploy::operation::dummy
