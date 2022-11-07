// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"

#include "mmdeploy/core/logger.h"

namespace mmdeploy::framework {

Result<Mat> MakeAvailableOnDevice(const Mat& src, const Device& device, Stream& stream) {
  if (src.device() == device) {
    return src;
  }

  Mat dst{src.height(), src.width(), src.pixel_format(), src.type(), device};
  OUTCOME_TRY(stream.Copy(src.buffer(), dst.buffer(), dst.byte_size()));

  // ! When the target device is different from stream's device (e.g. DtoH), insert a sync op as
  //   computation on dst won't be synchronized with stream
  if (device != stream.GetDevice()) {
    OUTCOME_TRY(stream.Wait());
  }

  return dst;
}

Result<Tensor> MakeAvailableOnDevice(const Tensor& src, const Device& device, Stream& stream) {
  if (src.device() == device) {
    return src;
  }

  TensorDesc desc{device, src.data_type(), src.shape(), src.name()};
  Tensor dst(desc);

  OUTCOME_TRY(stream.Copy(src.buffer(), dst.buffer(), src.byte_size()));

  // ! When the target device is different from stream's device (e.g. DtoH), insert a sync op as
  //   computation on dst won't be synchronized with stream
  if (device != stream.GetDevice()) {
    OUTCOME_TRY(stream.Wait());
  }

  return dst;
}

}  // namespace mmdeploy::framework
