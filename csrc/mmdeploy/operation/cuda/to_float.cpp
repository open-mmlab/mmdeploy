// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "mmdeploy/operation/vision.h"

namespace mmdeploy::operation::cuda {

template <typename From, typename To>
void Cast(const From* src, To* dst, size_t n, cudaStream_t stream);

class ToFloatImpl : public ToFloat {
 public:
  Result<void> apply(const Tensor& src, Tensor& dst) override {
    auto data_type = src.desc().data_type;
    if (data_type == DataType::kFLOAT) {
      dst = src;
      return success();
    }

    if (data_type == DataType::kINT8) {
      auto desc = src.desc();
      desc.data_type = DataType::kFLOAT;

      Tensor dst_tensor(desc);
      Cast(src.data<uint8_t>(), dst_tensor.data<float>(), src.size(),
           GetNative<cudaStream_t>(stream()));

      dst = std::move(dst_tensor);
      return success();
    }
    throw_exception(eNotSupported);
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ToFloat, (cuda, 0), [] { return std::make_unique<ToFloatImpl>(); });

}  // namespace mmdeploy::operation::cuda
