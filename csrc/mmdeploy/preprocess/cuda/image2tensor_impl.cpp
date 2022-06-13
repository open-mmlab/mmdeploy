// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/image2tensor.h"

namespace mmdeploy {
namespace cuda {

template <typename T>
void Transpose(const T* src, int height, int width, int channels, T* dst, cudaStream_t stream);

class ImageToTensorImpl final : public ::mmdeploy::ImageToTensorImpl {
 public:
  explicit ImageToTensorImpl(const Value& args) : ::mmdeploy::ImageToTensorImpl(args) {}

 protected:
  Result<Tensor> HWC2CHW(const Tensor& tensor) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device_, stream_));

    SyncOnScopeExit sync(stream_, src_tensor.buffer() != tensor.buffer(), src_tensor);

    auto h = tensor.shape(1);
    auto w = tensor.shape(2);
    auto c = tensor.shape(3);
    auto hw = h * w;

    Tensor dst_tensor(src_tensor.desc());
    dst_tensor.Reshape({1, c, h, w});

    auto stream = ::mmdeploy::GetNative<cudaStream_t>(stream_);

    if (DataType::kINT8 == tensor.data_type()) {
      auto input = src_tensor.data<uint8_t>();
      auto output = dst_tensor.data<uint8_t>();
      Transpose(input, (int)h, (int)w, (int)c, output, stream);
    } else if (DataType::kFLOAT == tensor.data_type()) {
      auto input = src_tensor.data<float>();
      auto output = dst_tensor.data<float>();
      Transpose(input, (int)h, (int)w, (int)c, output, stream);
    } else {
      assert(0);
    }
    return dst_tensor;
  }
};

class ImageToTensorImplCreator : public Creator<::mmdeploy::ImageToTensorImpl> {
 public:
  const char* GetName() const override { return "cuda"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& cfg) override { return std::make_unique<ImageToTensorImpl>(cfg); }
};

}  // namespace cuda
}  // namespace mmdeploy

using ::mmdeploy::ImageToTensorImpl;
using ::mmdeploy::cuda::ImageToTensorImplCreator;
REGISTER_MODULE(ImageToTensorImpl, ImageToTensorImplCreator);
