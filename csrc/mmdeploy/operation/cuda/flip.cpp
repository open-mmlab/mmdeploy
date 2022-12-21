// Copyright (c) OpenMMLab. All rights reserved.

#include "ppl/cv/cuda/flip.h"

#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/vision.h"

namespace mmdeploy::operation::cuda {

class FlipImpl : public Flip {
 public:
  using Flip::Flip;

  Result<void> apply(const Tensor& src, Tensor& dst) override {
    Tensor dst_tensor(src.desc());
    auto cuda_stream = GetNative<cudaStream_t>(stream());
    auto h = static_cast<int>(src.shape(1));
    auto w = static_cast<int>(src.shape(2));
    auto c = static_cast<int>(src.shape(3));
    ppl::common::RetCode ret;
    if (src.data_type() == DataType::kINT8) {
      auto input = src.data<uint8_t>();
      auto output = dst_tensor.data<uint8_t>();
      if (c == 1) {
        ret = ppl::cv::cuda::Flip<uint8_t, 1>(cuda_stream, h, w, w * c, input, w * c, output,
                                              flip_code_);
      } else if (c == 3) {
        ret = ppl::cv::cuda::Flip<uint8_t, 3>(cuda_stream, h, w, w * c, input, w * c, output,
                                              flip_code_);
      } else {
        ret = ppl::common::RC_UNSUPPORTED;
      }
    } else if (src.data_type() == DataType::kFLOAT) {
      auto input = src.data<float>();
      auto output = dst_tensor.data<float>();
      if (c == 1) {
        ret = ppl::cv::cuda::Flip<float, 1>(cuda_stream, h, w, w * c, input, w * c, output,
                                            flip_code_);
      } else if (c == 3) {
        ret = ppl::cv::cuda::Flip<float, 3>(cuda_stream, h, w, w * c, input, w * c, output,
                                            flip_code_);
      } else {
        ret = ppl::common::RC_UNSUPPORTED;
      }
    } else {
      MMDEPLOY_ERROR("unsupported data type {}", src.data_type());
      return Status(eNotSupported);
    }

    if (ret != 0) {
      return Status(eFail);
    }
    dst = std::move(dst_tensor);
    return success();
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Flip, (cuda, 0),
                               [](int flip_code) { return std::make_unique<FlipImpl>(flip_code); });

}  // namespace mmdeploy::operation::cuda
