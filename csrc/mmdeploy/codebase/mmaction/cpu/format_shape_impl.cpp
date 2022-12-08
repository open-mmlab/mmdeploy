// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmaction/format_shape.h"
#include "mmdeploy/core/utils/device_utils.h"

using namespace std;

namespace mmdeploy::mmaction::cpu {

class FormatShapeImpl : public FormatShapeOp {
 public:
  explicit FormatShapeImpl(std::string input_format) : FormatShapeOp(std::move(input_format)) {}

 protected:
  Result<void> apply(const std::vector<Tensor>& tensors, Tensor& output, int clip_len,
                     int num_clips) override {
    auto N = static_cast<int64_t>(tensors.size());
    auto H = tensors[0].shape(1);
    auto W = tensors[0].shape(2);
    auto C = tensors[0].shape(3);

    TensorDesc desc = {kHost, DataType::kFLOAT, {N, H, W, C}};
    Tensor imgs(desc);
    auto offset = 0UL;
    auto n_item = H * W * C;
    auto copy_size = n_item * sizeof(float);
    for (int i = 0; i < N; i++) {
      auto src_buffer = tensors[i].buffer();
      auto dst_buffer = imgs.buffer();
      OUTCOME_TRY(stream().Copy(src_buffer, dst_buffer, copy_size, 0, offset));
      offset += copy_size;
    }

    OUTCOME_TRY(stream().Wait());

    Tensor dst;
    if (input_format_ == "NCHW") {
      OUTCOME_TRY(dst, FormatNCHW(imgs, clip_len, num_clips));
    }
    if (input_format_ == "NCTHW") {
      OUTCOME_TRY(dst, FormatNCTHW(imgs, clip_len, num_clips));
    }
    TensorShape expand_dim = dst.shape();
    expand_dim.insert(expand_dim.begin(), 1);
    dst.Reshape(expand_dim);
    output = std::move(dst);

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
    auto L = clip_len;
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
    int ndim = shape.size();
    std::vector<int> dst_strides(ndim);
    std::vector<int> src_strides(ndim);
    dst_strides[ndim - 1] = src_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
      dst_strides[i] = dst_strides[i + 1] * shape[i + 1];
      src_strides[i] = src_strides[i + 1] * src_dims[i + 1];
    }
    std::vector<int> tmp(ndim);
    for (int i = 0; i < ndim; i++) {
      tmp[i] = src_strides[permutation[i]];
    }
    src_strides.swap(tmp);
    std::vector<int> coord(ndim, 0);
    auto dst_data = dst.data<float>();
    auto src_data = src.data<float>();

    int i;
    do {
      dst_data[0] = src_data[0];
      for (i = ndim - 1; i >= 0; i--) {
        if (++coord[i] == shape[i]) {
          coord[i] = 0;
          dst_data -= (shape[i] - 1) * dst_strides[i];
          src_data -= (shape[i] - 1) * src_strides[i];
        } else {
          dst_data += dst_strides[i];
          src_data += src_strides[i];
          break;
        }
      }
    } while (i >= 0);
    return dst;
  }

  constexpr static Device kHost{0, 0};
};

MMDEPLOY_REGISTER_FACTORY_FUNC(FormatShapeOp, (cpu, 0), [](std::string input_format) {
  return std::make_unique<FormatShapeImpl>(std::move(input_format));
});

}  // namespace mmdeploy::mmaction::cpu
