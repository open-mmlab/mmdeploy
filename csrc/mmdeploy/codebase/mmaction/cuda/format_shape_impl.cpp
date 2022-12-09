// Copyright (c) OpenMMLab. All rights reserved.

#include "cuda_runtime.h"
#include "mmdeploy/codebase/mmaction/format_shape.h"
#include "mmdeploy/core/utils/device_utils.h"

using namespace std;

namespace mmdeploy::mmaction::cuda {

template <typename T>
void Transpose(const T* src, const int* src_strides, T* dst, const int* dst_strides, int ndim,
               int total, cudaStream_t stream);

class FormatShapeImpl : public FormatShapeOp {
 public:
  explicit FormatShapeImpl(std::string input_format) : FormatShapeOp(std::move(input_format)) {}

 protected:
  Result<void> apply(const std::vector<Tensor>& inputs, Tensor& output, int clip_len,
                     int num_clips) override {
    auto N = static_cast<int64_t>(inputs.size());
    auto H = inputs[0].shape(1);
    auto W = inputs[0].shape(2);
    auto C = inputs[0].shape(3);

    auto t0 = std::chrono::high_resolution_clock::now();
    TensorDesc desc = {device_, DataType::kFLOAT, {N, H, W, C}};
    Tensor imgs(desc);
    int offset = 0;
    int n_item = H * W * C;
    int copy_size = n_item * sizeof(float);
    for (int i = 0; i < N; i++) {
      auto src_buffer = inputs[i].buffer();
      auto dst_buffer = imgs.buffer();
      OUTCOME_TRY(stream().Copy(src_buffer, dst_buffer, copy_size, 0, offset));
      offset += copy_size;
    }

    // Tensor dst;
    if (input_format_ == "NCHW") {
      OUTCOME_TRY(output, FormatNCHW(imgs, clip_len, num_clips));
    }
    if (input_format_ == "NCTHW") {
      OUTCOME_TRY(output, FormatNCTHW(imgs, clip_len, num_clips));
    }
    TensorShape expand_dim = output.shape();
    expand_dim.insert(expand_dim.begin(), 1);
    output.Reshape(expand_dim);

    return success();
  }

  Result<Tensor> FormatNCHW(Tensor& src, int clip_len, int num_clips) {
    auto N = src.shape(0);
    auto H = src.shape(1);
    auto W = src.shape(2);
    auto C = src.shape(3);
    return Transpose(src, {N, H, W, C}, {0, 3, 1, 2});
  };

  Result<Tensor> FormatNCTHW(Tensor& src, int clip_len, int num_clips) {
    auto N = src.shape(0);
    auto H = src.shape(1);
    auto W = src.shape(2);
    auto C = src.shape(3);
    int L = clip_len;
    if (N % L != 0) {
      return Status(eInvalidArgument);
    }
    int M = N / L;
    src.Reshape({M, L, H, W, C});

    return Transpose(src, {M, L, H, W, C}, {0, 4, 1, 2, 3});
  };

  Result<Tensor> Transpose(Tensor& src, const TensorShape& src_dims,
                           const std::vector<int>& permutation) {
    Tensor dst(src.desc());
    TensorShape shape(src.shape().size());
    for (int i = 0; i < shape.size(); i++) {
      shape[i] = src.shape(permutation[i]);
    }
    dst.Reshape(shape);

    auto ndim = src_dims.size();
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
    OUTCOME_TRY(stream().Copy(src_strides.data(), _src_strides));
    OUTCOME_TRY(stream().Copy(dst_strides.data(), _dst_strides));

    ::mmdeploy::mmaction::cuda::Transpose(src.data<float>(), GetNative<int*>(_src_strides),
                                          dst.data<float>(), GetNative<int*>(_dst_strides), ndim,
                                          src.size(), (cudaStream_t)stream().GetNative());
    return dst;
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(FormatShapeOp, (cuda, 0), [](std::string input_format) {
  return std::make_unique<FormatShapeImpl>(std::move(input_format));
});

}  // namespace mmdeploy::mmaction::cuda
