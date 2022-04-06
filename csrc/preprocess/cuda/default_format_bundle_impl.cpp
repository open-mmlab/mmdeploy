// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "core/utils/device_utils.h"
#include "preprocess/transform/default_format_bundle.h"

namespace mmdeploy {
namespace cuda {

template <int channels>
void CastToFloat(const uint8_t* src, int height, int width, float* dst, cudaStream_t stream);

template <typename T>
void Transpose(const T* src, int height, int width, int channels, T* dst, cudaStream_t stream);

class DefaultFormatBundleImpl final : public ::mmdeploy::DefaultFormatBundleImpl {
 public:
  explicit DefaultFormatBundleImpl(const Value& args) : ::mmdeploy::DefaultFormatBundleImpl(args) {}

 protected:
  Result<Tensor> ToFloat32(const Tensor& tensor, const bool& img_to_float) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device_, stream_));

    SyncOnScopeExit sync(stream_, src_tensor.buffer() != tensor.buffer(), src_tensor);

    auto data_type = src_tensor.data_type();
    auto h = tensor.shape(1);
    auto w = tensor.shape(2);
    auto c = tensor.shape(3);
    auto stream = ::mmdeploy::GetNative<cudaStream_t>(stream_);

    if (img_to_float && data_type == DataType::kINT8) {
      TensorDesc desc{device_, DataType::kFLOAT, tensor.shape(), ""};
      Tensor dst_tensor{desc};
      if (c == 3) {
        CastToFloat<3>(src_tensor.data<uint8_t>(), h, w, dst_tensor.data<float>(), stream);
      } else if (c == 1) {
        CastToFloat<1>(src_tensor.data<uint8_t>(), h, w, dst_tensor.data<float>(), stream);
      } else {
        MMDEPLOY_ERROR("channel num: unsupported channel num {}", c);
        return Status(eNotSupported);
      }
      return dst_tensor;
    }
    return src_tensor;
  }

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

class DefaultFormatBundleImplCreator : public Creator<::mmdeploy::DefaultFormatBundleImpl> {
 public:
  const char* GetName() const override { return "cuda"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& cfg) override {
    return std::make_unique<DefaultFormatBundleImpl>(cfg);
  }
};

}  // namespace cuda
}  // namespace mmdeploy

using ::mmdeploy::DefaultFormatBundleImpl;
using ::mmdeploy::cuda::DefaultFormatBundleImplCreator;
REGISTER_MODULE(DefaultFormatBundleImpl, DefaultFormatBundleImplCreator);
