// Copyright (c) OpenMMLab. All rights reserved.

#include "transform_utils.h"
namespace mmdeploy {

Result<Mat> MakeAvailableOnDevice(const Mat& src, const Device& device, Stream& stream) {
  if (src.device() == device) {
    return src;
  }

  Mat dst{src.height(), src.width(), src.pixel_format(), src.type(), device};
  OUTCOME_TRY(stream.Copy(src.buffer(), dst.buffer(), dst.byte_size()));
  return dst;
}

Result<Tensor> MakeAvailableOnDevice(const Tensor& src, const Device& device, Stream& stream) {
  if (src.device() == device) {
    return src;
  }

  TensorDesc desc{device, src.data_type(), src.shape(), src.name()};
  Tensor dst(desc);

  OUTCOME_TRY(stream.Copy(src.buffer(), dst.buffer(), src.byte_size()));

  return dst;
}

}  // namespace mmdeploy
