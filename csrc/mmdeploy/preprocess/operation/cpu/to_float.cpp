// Copyright (c) OpenMMLab. All rights reserved.

#include <map>

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class ToFloatImpl : public ToFloat {
 public:
  using ToFloat::ToFloat;

  Result<Tensor> to_float(const Tensor& tensor) override {
    auto data_type = tensor.desc().data_type;
    if (data_type == DataType::kFLOAT) {
      return tensor;
    }

    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device(), stream()));
    SyncOnScopeExit(stream(), src_tensor.buffer() != tensor.buffer(), src_tensor);

    if (data_type == DataType::kINT8) {
      const auto size = tensor.size();
      if (size > std::numeric_limits<int>::max()) {
        throw_exception(eNotSupported);
      }
      cv::Mat uint8_mat(1, static_cast<int>(size), CV_8U, src_tensor.data());

      auto desc = src_tensor.desc();
      desc.data_type = DataType::kFLOAT;
      Tensor dst_tensor(desc);

      cv::Mat float_mat(1, static_cast<int>(size), CV_32F, dst_tensor.data());
      uint8_mat.convertTo(float_mat, CV_32F);

      return dst_tensor;
    }
    throw_exception(eNotSupported);
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ToFloat, (cpu, 0), [](const Context& context) {
  return std::make_unique<ToFloatImpl>(context);
});

}  // namespace mmdeploy::operation::cpu
