// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/operation/vision.h"

namespace mmdeploy::operation::cuda {

template <typename From, typename To>
void Cast(const From* src, To* dst, size_t n, cudaStream_t stream);

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
      auto desc = src_tensor.desc();
      desc.data_type = DataType::kFLOAT;

      Tensor dst_tensor(desc);
      Cast(src_tensor.data<uint8_t>(), dst_tensor.data<float>(), src_tensor.size(),
           GetNative<cudaStream_t>(stream()));

      return dst_tensor;
    }
    throw_exception(eNotSupported);
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ToFloat, (cuda, 0), [](const Context& context) {
  return std::make_unique<ToFloatImpl>(context);
});

}  // namespace mmdeploy::operation::cuda
