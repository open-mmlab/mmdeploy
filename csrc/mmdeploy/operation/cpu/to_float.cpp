// Copyright (c) OpenMMLab. All rights reserved.

#include <map>

#include "mmdeploy/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class ToFloatImpl : public ToFloat {
 public:
  Result<void> apply(const Tensor& src, Tensor& dst) override {
    auto data_type = src.desc().data_type;
    if (data_type == DataType::kFLOAT) {
      dst = src;
      return success();
    }

    if (data_type == DataType::kINT8) {
      const auto size = src.size();
      if (size > std::numeric_limits<int>::max()) {
        throw_exception(eNotSupported);
      }
      cv::Mat uint8_mat(1, static_cast<int>(size), CV_8U, const_cast<void*>(src.data()));

      auto desc = src.desc();
      desc.data_type = DataType::kFLOAT;
      Tensor dst_tensor(desc);

      cv::Mat float_mat(1, static_cast<int>(size), CV_32F, dst_tensor.data());
      uint8_mat.convertTo(float_mat, CV_32F);

      dst = std::move(dst_tensor);
      return success();
    }
    throw_exception(eNotSupported);
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ToFloat, (cpu, 0), []() { return std::make_unique<ToFloatImpl>(); });

}  // namespace mmdeploy::operation::cpu
