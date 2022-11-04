//
// Created by zhangli on 11/3/22.
//

#ifndef MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_RESIZE_H_
#define MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_RESIZE_H_

#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/preprocess/operation/operation.h"

namespace mmdeploy::operation {

class MMDEPLOY_API ToBGR : public Operation {
 public:
  using Operation::Operation;

  virtual Result<Tensor> to_bgr(const Mat& img) = 0;
};
MMDEPLOY_DECLARE_REGISTRY(ToBGR, unique_ptr<ToBGR>(const Context& context));

class ToGray : public Operation {
 public:
  using Operation::Operation;

  virtual Result<Tensor> to_gray(const Mat& img) = 0;
};
MMDEPLOY_DECLARE_REGISTRY(ToGray, unique_ptr<ToGray>(const Context& context));

// resize in HWC format
class Resize : public Operation {
 public:
  explicit Resize(const string_view& interp, const Context& context)
      : Operation(context), interp_(interp) {}

  virtual Result<Tensor> resize(const Tensor& img, int dst_h, int dst_w) = 0;

 protected:
  std::string interp_;
};
MMDEPLOY_DECLARE_REGISTRY(Resize,
                          unique_ptr<Resize>(const string_view& interp, const Context& context));

// pad in HWC format
class Pad : public Operation {
 public:
  using Operation::Operation;

  virtual Result<Tensor> pad(const Tensor& tensor, int top, int left, int bottom, int right) = 0;
};
MMDEPLOY_DECLARE_REGISTRY(Pad, unique_ptr<Pad>(const string_view& border_type, float pad_val,
                                               const Context& context));

// uint8 to float
class ToFloat : public Operation {
 public:
  using Operation::Operation;

  virtual Result<Tensor> to_float(const Tensor& tensor) = 0;
};
MMDEPLOY_DECLARE_REGISTRY(ToFloat, unique_ptr<ToFloat>(const Context& context));

class HWC2CHW : public Operation {
 public:
  using Operation::Operation;

  virtual Result<Tensor> hwc2chw(const Tensor& tensor) = 0;
};
MMDEPLOY_DECLARE_REGISTRY(HWC2CHW, unique_ptr<HWC2CHW>(const Context& context));

// normalize in HWC format
class Normalize : public Operation {
 public:
  struct Param {
    std::vector<float> mean;
    std::vector<float> std;
    bool to_rgb;
  };

  using Operation::Operation;

  virtual Result<Tensor> normalize(const Tensor& img) = 0;
};
MMDEPLOY_DECLARE_REGISTRY(Normalize, unique_ptr<Normalize>(const Normalize::Param& param,
                                                           const Context& context));

// crop in HWC format
class Crop : public Operation {
 public:
  using Operation::Operation;

  virtual Result<Tensor> crop(const Tensor& tensor, int top, int left, int bottom, int right) = 0;
};
MMDEPLOY_DECLARE_REGISTRY(Crop, unique_ptr<Crop>(const Context& context));

// TODO: warp affine

}  // namespace mmdeploy::operation

#endif  // MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_RESIZE_H_
