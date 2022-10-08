// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmaction/format_shape.h"
#include "mmdeploy/core/utils/device_utils.h"

using namespace std;

namespace mmdeploy {
namespace cpu {

class IndexHelper {
 public:
  IndexHelper(const std::vector<int>& src_dims, const std::vector<int>& permutation) {
    assert(permutation.size() == src_dims.size());

    ndims_ = src_dims.size();
    permutation_ = permutation;
    src_stride_.resize(ndims_);
    src_stride_[ndims_ - 1] = 1;
    for (int i = ndims_ - 2; i >= 0; i--) {
      src_stride_[i] = src_stride_[i + 1] * src_dims[i + 1];
    }

    std::vector<int> dst_dims(ndims_);
    for (int i = 0; i < permutation.size(); i++) {
      dst_dims[i] = src_dims[permutation[i]];
    }
    dst_stride_.resize(ndims_);
    dst_stride_[ndims_ - 1] = 1;
    for (int i = ndims_ - 2; i >= 0; i--) {
      dst_stride_[i] = dst_stride_[i + 1] * dst_dims[i + 1];
    }
  }

  int GetSrcOffset(int offset) {
    // dst offset -> dst index -> src index -> src offset

    // dst index
    int remaining = offset;
    std::vector<int> dst_index(ndims_);
    std::vector<int> src_index(ndims_);
    for (int i = 0; i < ndims_; i++) {
      int idx = remaining / dst_stride_[i];
      dst_index[i] = idx;
      remaining -= idx * dst_stride_[i];
    }

    // src index
    for (int i = 0; i < ndims_; i++) {
      src_index[permutation_[i]] = dst_index[i];
    }

    // src offset
    int src_offset = 0;
    for (int i = 0; i < ndims_; i++) {
      src_offset += src_index[i] * src_stride_[i];
    }

    return src_offset;
  }

 private:
  vector<int> permutation_;
  vector<int> src_stride_;
  vector<int> dst_stride_;
  int ndims_;
};

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

    IndexHelper helper(src_dims, permutation);
    auto dst_data = dst.data<float>();
    auto src_data = src.data<float>();
    for (int i = 0; i < dst.size(); i++) {
      int src_offset = helper.GetSrcOffset(i);
      dst_data[i] = src_data[src_offset];
    }
    return dst;
  }

  constexpr static Device kHost{0, 0};
};

class FormatShapeImplCreator : public Creator<::mmdeploy::FormatShapeImpl> {
 public:
  const char* GetName() const override { return "cpu"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override { return make_unique<FormatShapeImpl>(args); }
};

}  // namespace cpu
}  // namespace mmdeploy

using ::mmdeploy::FormatShapeImpl;
using ::mmdeploy::cpu::FormatShapeImplCreator;
REGISTER_MODULE(FormatShapeImpl, FormatShapeImplCreator);
