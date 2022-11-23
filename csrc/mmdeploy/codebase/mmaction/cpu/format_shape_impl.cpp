// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmaction/format_shape.h"
#include "mmdeploy/core/utils/device_utils.h"

using namespace std;

namespace mmdeploy {
namespace cpu {

class FormatShapeImpl : public ::mmdeploy::FormatShapeImpl {
 public:
  explicit FormatShapeImpl(const Value& args) : ::mmdeploy::FormatShapeImpl(args) {}

 protected:
  Result<Tensor> Format(const std::vector<Tensor>& tensors, int clip_len, int num_clips) {
    int N = tensors.size();
    int H = tensors[0].shape(1);
    int W = tensors[0].shape(2);
    int C = tensors[0].shape(3);

    std::vector<Tensor> host_tensors;
    host_tensors.reserve(N);
    for (int i = 0; i < N; i++) {
      OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensors[i], kHost, stream_));
      host_tensors.push_back(std::move(src_tensor));
    }
    OUTCOME_TRY(stream_.Wait());

    TensorDesc desc = {kHost, DataType::kFLOAT, {N, H, W, C}};
    Tensor imgs(desc);
    int offset = 0;
    int n_item = H * W * C;
    int copy_size = n_item * sizeof(float);
    for (int i = 0; i < N; i++) {
      auto src_buffer = host_tensors[i].buffer();
      auto dst_buffer = imgs.buffer();
      OUTCOME_TRY(stream_.Copy(src_buffer, dst_buffer, copy_size, 0, offset));
      offset += copy_size;
    }
    OUTCOME_TRY(stream_.Wait());

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

MMDEPLOY_REGISTER_TRANSFORM_IMPL(::mmdeploy::FormatShapeImpl, (cpu, 0), FormatShapeImpl);

}  // namespace cpu
}  // namespace mmdeploy
