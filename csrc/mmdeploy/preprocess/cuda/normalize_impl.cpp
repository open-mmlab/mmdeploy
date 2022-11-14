// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/transform/normalize.h"

using namespace std;

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

    TensorDesc dst_desc{device_, DataType::kFLOAT, src_desc.shape, src_desc.name};
    Tensor dst_tensor{dst_desc};
    auto output = dst_tensor.data<float>();
    auto stream = ::mmdeploy::GetNative<cudaStream_t>(stream_);

    if (DataType::kINT8 == src_desc.data_type) {
      auto input = src_tensor.data<uint8_t>();
      if (3 == c) {
        Normalize<uint8_t, 3>(input, h, w, stride, output, arg_.mean.data(), arg_.std.data(),
                              arg_.to_rgb, stream);
      } else if (1 == c) {
        Normalize<uint8_t, 1>(input, h, w, stride, output, arg_.mean.data(), arg_.std.data(),
                              arg_.to_rgb, stream);
      } else {
        MMDEPLOY_ERROR("unsupported channels {}", c);
        return Status(eNotSupported);
      }
    } else if (DataType::kFLOAT == src_desc.data_type) {
      auto input = src_tensor.data<float>();
      if (3 == c) {
        Normalize<float, 3>(input, h, w, stride, output, arg_.mean.data(), arg_.std.data(),
                            arg_.to_rgb, stream);
      } else if (1 == c) {
        Normalize<float, 1>(input, h, w, stride, output, arg_.mean.data(), arg_.std.data(),
                            arg_.to_rgb, stream);
      } else {
        MMDEPLOY_ERROR("unsupported channels {}", c);
        return Status(eNotSupported);
      }
    } else {
      MMDEPLOY_ERROR("unsupported data type {}", src_desc.data_type);
      assert(0);
      return Status(eNotSupported);
    }
    return dst_tensor;
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
