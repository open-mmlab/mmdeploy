// Copyright (c) OpenMMLab. All rights reserved.

#include "cuda_runtime.h"
#include "mmdeploy/codebase/mmaction/format_shape.h"
#include "mmdeploy/core/utils/device_utils.h"

using namespace std;

namespace mmdeploy {
namespace cuda {

template <typename T>
void Transpose(const T* src, const int* src_strides, T* dst, const int* dst_strides, int ndim,
               int total, cudaStream_t stream);

class FormatShapeImpl : public ::mmdeploy::FormatShapeImpl {
 public:
  explicit FormatShapeImpl(const Value& args) : ::mmdeploy::FormatShapeImpl(args) {}

 protected:
  Result<Tensor> Format(const std::vector<Tensor>& tensors, int clip_len, int num_clips) {
    int N = tensors.size();
    int H = tensors[0].shape(1);
    int W = tensors[0].shape(2);
    int C = tensors[0].shape(3);

    auto t0 = std::chrono::high_resolution_clock::now();
    TensorDesc desc = {device_, DataType::kFLOAT, {N, H, W, C}};
    Tensor imgs(desc);
    int offset = 0;
    int n_item = H * W * C;
    int copy_size = n_item * sizeof(float);
    for (int i = 0; i < N; i++) {
      auto src_buffer = tensors[i].buffer();
      auto dst_buffer = imgs.buffer();
      OUTCOME_TRY(stream_.Copy(src_buffer, dst_buffer, copy_size, 0, offset));
      offset += copy_size;
    }

    Tensor dst;
    if (arg_.input_format == "NCHW") {
      OUTCOME_TRY(dst, FormatNCHW(imgs, clip_len, num_clips));
    }
    if (arg_.input_format == "NCTHW") {
      OUTCOME_TRY(dst, FormatNCTHW(imgs, clip_len, num_clips));
    }
    TensorShape expand_dim = dst.shape();
    expand_dim.insert(expand_dim.begin(), 1);
    dst.Reshape(expand_dim);

    return dst;
  }

  Result<Tensor> FormatNCHW(Tensor& src, int clip_len, int num_clips) {
    int N = src.shape(0);
    int H = src.shape(1);
    int W = src.shape(2);
    int C = src.shape(3);
    return Transpose(src, {N, H, W, C}, {0, 3, 1, 2});
  };

  Result<Tensor> FormatNCTHW(Tensor& src, int clip_len, int num_clips) {
    int N = src.shape(0);
    int H = src.shape(1);
    int W = src.shape(2);
    int C = src.shape(3);
    int L = clip_len;
    if (N % L != 0) {
      return Status(eInvalidArgument);
    }
    int M = N / L;
    src.Reshape({M, L, H, W, C});

    return Transpose(src, {M, L, H, W, C}, {0, 4, 1, 2, 3});
  };

  Result<Tensor> Transpose(Tensor& src, const std::vector<int>& src_dims,
                           const std::vector<int>& permutation) {
    Tensor dst(src.desc());
    TensorShape shape(src.shape().size());
    for (int i = 0; i < shape.size(); i++) {
      shape[i] = src.shape(permutation[i]);
    }
    dst.Reshape(shape);

    int ndim = src_dims.size();
    std::vector<int> dst_dims(ndim);
    for (int i = 0; i < ndim; i++) {
      dst_dims[i] = src_dims[permutation[i]];
    }

    std::vector<int> src_strides(ndim);
    std::vector<int> dst_strides(ndim);
    std::vector<int> buffer(ndim);
    buffer.back() = 1;
    dst_strides.back() = 1;
    for (int i = ndim - 1; i > 0; i--) {
      buffer[i - 1] = buffer[i] * src_dims[i];
      dst_strides[i - 1] = dst_strides[i] * dst_dims[i];
    }
    for (int i = 0; i < ndim; ++i) {
      src_strides[i] = buffer[permutation[i]];
    }

    Buffer _src_strides(Device("cuda"), sizeof(int) * ndim);
    Buffer _dst_strides(Device("cuda"), sizeof(int) * ndim);
    OUTCOME_TRY(stream_.Copy(src_strides.data(), _src_strides));
    OUTCOME_TRY(stream_.Copy(dst_strides.data(), _dst_strides));

    ::mmdeploy::cuda::Transpose(src.data<float>(), GetNative<int*>(_src_strides), dst.data<float>(),
                                GetNative<int*>(_dst_strides), ndim, src.size(),
                                (cudaStream_t)stream_.GetNative());
    return dst;
  }
};

class FormatShapeImplCreator : public Creator<::mmdeploy::FormatShapeImpl> {
 public:
  const char* GetName() const override { return "cuda"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override { return make_unique<FormatShapeImpl>(args); }
};

}  // namespace cuda
}  // namespace mmdeploy

using ::mmdeploy::FormatShapeImpl;
using ::mmdeploy::cuda::FormatShapeImplCreator;
REGISTER_MODULE(FormatShapeImpl, FormatShapeImplCreator);
