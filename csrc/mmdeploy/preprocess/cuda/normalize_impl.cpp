// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/transform/normalize.h"
#include "ppl/cv/cuda/cvtcolor.h"

using namespace std;
using namespace ppl::cv::cuda;

namespace mmdeploy::cuda {

template <typename T, int channels>
void Normalize(const T* src, int height, int width, int stride, float* output, const float* mean,
               const float* std, bool to_rgb, cudaStream_t stream);

class NormalizeImpl : public ::mmdeploy::NormalizeImpl {
 public:
  explicit NormalizeImpl(const Value& args) : ::mmdeploy::NormalizeImpl(args) {}

 protected:
  Result<Tensor> NormalizeImage(const Tensor& tensor) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device_, stream_));

    SyncOnScopeExit sync(stream_, src_tensor.buffer() != tensor.buffer(), src_tensor);

    auto src_desc = src_tensor.desc();
    int h = (int)src_desc.shape[1];
    int w = (int)src_desc.shape[2];
    int c = (int)src_desc.shape[3];
    int stride = w * c;
    auto stream = ::mmdeploy::GetNative<cudaStream_t>(stream_);

    if (!arg_.to_float) {
      if (!arg_.to_rgb) {
        return tensor;
      }
      TensorDesc dst_desc{device_, DataType::kINT8, src_desc.shape, src_desc.name};
      Tensor dst_tensor{dst_desc};
      RGB2BGR<uint8_t>(stream, h, w, stride, tensor.data<uint8_t>(), stride,
                       dst_tensor.data<uint8_t>());
      return dst_tensor;
    } else {
      TensorDesc dst_desc{device_, DataType::kFLOAT, src_desc.shape, src_desc.name};
      Tensor dst_tensor{dst_desc};
      auto output = dst_tensor.data<float>();

      if (DataType::kINT8 == src_desc.data_type) {
        Dispatch<uint8_t>(src_tensor.data<uint8_t>(), h, w, c, stride, output, arg_.mean.data(),
                          arg_.std.data(), arg_.to_rgb, stream).value();
      } else if (DataType::kFLOAT == src_desc.data_type) {
        Dispatch<float>(src_tensor.data<float>(), h, w, c, stride, output, arg_.mean.data(),
                        arg_.std.data(), arg_.to_rgb, stream).value();
      } else {
        MMDEPLOY_ERROR("unsupported data type {}", src_desc.data_type);
        return Status(eNotSupported);
      }
      return dst_tensor;
    }
  }

  template <typename T>
  Result<void> Dispatch(const T* input, int height, int width, int channel, int stride,
                        float* output, const float* mean, const float* std, bool to_rgb,
                        cudaStream_t stream) {
    if (3 == channel) {
      Normalize<T, 3>(input, height, width, stride, output, arg_.mean.data(), arg_.std.data(),
                      arg_.to_rgb, stream);
    } else if (1 == channel) {
      Normalize<T, 1>(input, height, width, stride, output, arg_.mean.data(), arg_.std.data(),
                      arg_.to_rgb, stream);
    } else {
      MMDEPLOY_ERROR("unsupported channels {}", channel);
      return Status(eNotSupported);
    }
    return success();
  }
};

class NormalizeImplCreator : public Creator<::mmdeploy::NormalizeImpl> {
 public:
  const char* GetName() const override { return "cuda"; }
  int GetVersion() const override { return 1; }
  std::unique_ptr<::mmdeploy::NormalizeImpl> Create(const Value& args) override {
    return make_unique<NormalizeImpl>(args);
  }
};

}  // namespace mmdeploy::cuda

using mmdeploy::NormalizeImpl;
using mmdeploy::cuda::NormalizeImplCreator;
REGISTER_MODULE(NormalizeImpl, NormalizeImplCreator);
