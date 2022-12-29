// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_RESIZE_H_
#define MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_RESIZE_H_

#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/operation/operation.h"

namespace mmdeploy::operation {

class CvtColor : public Operation {
 public:
  virtual Result<void> apply(const Mat& src, Mat& dst, PixelFormat dst_fmt) = 0;
};
MMDEPLOY_DECLARE_REGISTRY(CvtColor, unique_ptr<CvtColor>());

// resize in HWC format
class Resize : public Operation {
 public:
  virtual Result<void> apply(const Tensor& src, Tensor& dst, int dst_h, int dst_w) = 0;
};
MMDEPLOY_DECLARE_REGISTRY(Resize, unique_ptr<Resize>(const string_view& interp));

// pad in HWC format
class Pad : public Operation {
 public:
  virtual Result<void> apply(const Tensor& src, Tensor& dst, int top, int left, int bottom,
                             int right) = 0;
};
MMDEPLOY_DECLARE_REGISTRY(Pad, unique_ptr<Pad>(const string_view& border_type, float pad_val));

// uint8 to float
class ToFloat : public Operation {
 public:
  virtual Result<void> apply(const Tensor& src, Tensor& dst) = 0;
};
MMDEPLOY_DECLARE_REGISTRY(ToFloat, unique_ptr<ToFloat>());

class HWC2CHW : public Operation {
 public:
  virtual Result<void> apply(const Tensor& src, Tensor& dst) = 0;
};
MMDEPLOY_DECLARE_REGISTRY(HWC2CHW, unique_ptr<HWC2CHW>());

// normalize in HWC format
class Normalize : public Operation {
 public:
  struct Param {
    std::vector<float> mean;
    std::vector<float> std;
    bool to_rgb;
  };

  virtual Result<void> apply(const Tensor& src, Tensor& dst) = 0;
};
MMDEPLOY_DECLARE_REGISTRY(Normalize, unique_ptr<Normalize>(const Normalize::Param& param));

// crop in HWC format
class Crop : public Operation {
 public:
  virtual Result<void> apply(const Tensor& src, Tensor& dst, int top, int left, int bottom,
                             int right) = 0;
};
MMDEPLOY_DECLARE_REGISTRY(Crop, unique_ptr<Crop>());

class Flip : public Operation {
 public:
  explicit Flip(int flip_code) : flip_code_(flip_code) {}

  virtual Result<void> apply(const Tensor& src, Tensor& dst) = 0;

 protected:
  int flip_code_;
};
MMDEPLOY_DECLARE_REGISTRY(Flip, unique_ptr<Flip>(int flip_code));

// 2x3 OpenCV affine matrix, row major
class WarpAffine : public Operation {
 public:
  virtual Result<void> apply(const Tensor& src, Tensor& dst, const float affine_matrix[6],
                             int dst_h, int dst_w) = 0;
};
MMDEPLOY_DECLARE_REGISTRY(WarpAffine, unique_ptr<WarpAffine>(const string_view& interp));

}  // namespace mmdeploy::operation

#endif  // MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_RESIZE_H_
