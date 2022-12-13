// Copyright (c) OpenMMLab. All rights reserved.

#include <opencv2/core.hpp>

#include "mmdeploy/codebase/mmedit/mmedit.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"

namespace mmdeploy::mmedit {

class TensorToImg : public MMEdit {
 public:
  explicit TensorToImg(const Value& cfg) : MMEdit(cfg) {}

  Result<Value> operator()(const Value& input) {
    auto upscale = input["output"].get<Tensor>();
    OUTCOME_TRY(auto upscale_cpu, MakeAvailableOnDevice(upscale, kHOST, stream()));
    OUTCOME_TRY(stream().Wait());
    if (upscale.shape().size() == 4 && upscale.data_type() == DataType::kFLOAT) {
      auto channels = static_cast<int>(upscale.shape(1));
      auto height = static_cast<int>(upscale.shape(2));
      auto width = static_cast<int>(upscale.shape(3));
      // TODO: handle BGR <-> RGB conversion
      OUTCOME_TRY(auto format, ChannelsToFormat(channels));
      Mat mat(height, width, format, DataType::kINT8, kHOST);
      cv::Mat_<float> mat_chw(channels, height * width, upscale_cpu.data<float>());
      cv::Mat mat_hwc(height * width, channels, CV_32F);
      cv::transpose(mat_chw, mat_hwc);
      cv::Mat rescale_uint8(height, width, CV_8UC(channels), mat.data<uint8_t>());
      mat_hwc = mat_hwc.reshape(channels, height);
      // convert has saturate_cast inside
      mat_hwc.convertTo(rescale_uint8, CV_8UC(channels), 255.f);
      return mat;
    } else {
      MMDEPLOY_ERROR("unsupported `output` tensor, shape: {}, dtype: {}", upscale.shape(),
                     (int)upscale.data_type());
      return Status(eNotSupported);
    }
  }

 protected:
  static Result<PixelFormat> ChannelsToFormat(int channels) {
    switch (channels) {
      case 1:
        return PixelFormat::kGRAYSCALE;
      case 3:
        return PixelFormat::kRGB;
      default:
        return Status(eNotSupported);
    }
  }

  static constexpr const Device kHOST{0, 0};
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMEdit, TensorToImg);

}  // namespace mmdeploy::mmedit
