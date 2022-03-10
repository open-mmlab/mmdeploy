// Copyright (c) OpenMMLab. All rights reserved.

#include "device_utils.h"
#include "core/logger.h"

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

ForceSync::~ForceSync() {
  if (active_ && stream_) {
    if (!stream_.Wait()) {
      MMDEPLOY_ERROR("Implicit stream synchronization failed.");
    } else {
      MMDEPLOY_INFO("Implicit stream synchronization succeeded.");
    }
  }
}

}  // namespace mmdeploy
