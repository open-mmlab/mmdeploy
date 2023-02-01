// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/cuda/permute.h"

#include <cuda_runtime.h>

#include "mmdeploy/operation/vision.h"

namespace mmdeploy::operation::cuda {

namespace impl {
template <typename T>
void Permute(const T* src, const TensorStride& src_strides, T* dst, const TensorStride& dst_strides,
             int ndim, int total, cudaStream_t stream);
}

class PermuteImpl : public Permute {
 public:
  explicit PermuteImpl() {}

  Result<void> apply(const Tensor& src, Tensor& dst, const std::vector<int>& axes) override {
    int ndim = src.shape().size();
    if (ndim != axes.size()) {
      MMDEPLOY_ERROR("The size of axes should be equal of src, {} vs {}", axes.size(), ndim);
      return Status(eInvalidArgument);
    }
    if (ndim > MAX_PERMUTE_DIM) {
      MMDEPLOY_ERROR("Only support ndim < {}", MAX_PERMUTE_DIM);
      return Status(eInvalidArgument);
    }
    std::vector<int> axes_vis(ndim, 0);
    for (const auto& x : axes) {
      if (x < 0 || x >= ndim || axes_vis[x]) {
        MMDEPLOY_ERROR("Invalid axes");
        return Status(eInvalidArgument);
      }
      axes_vis[x] = 1;
    }

    Tensor dst_tensor(src.desc());
    auto src_dims = src.shape();
    TensorShape dst_dims(ndim);
    for (int i = 0; i < src_dims.size(); i++) {
      dst_dims[i] = src_dims[axes[i]];
    }
    dst_tensor.Reshape(dst_dims);

    TensorStride dst_strides;
    TensorStride src_strides;

    dst_strides[ndim - 1] = src_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
      dst_strides[i] = dst_strides[i + 1] * dst_dims[i + 1];
      src_strides[i] = src_strides[i + 1] * src_dims[i + 1];
    }

    TensorStride tmp;
    for (int i = 0; i < ndim; i++) {
      tmp[i] = src_strides[axes[i]];
    }
    src_strides = tmp;

    if (src.data_type() == DataType::kINT8) {
      OUTCOME_TRY(PermuteDispatch<uint8_t>(src, dst_tensor, src_strides, dst_strides));
    } else if (src.data_type() == DataType::kFLOAT) {
      OUTCOME_TRY(PermuteDispatch<float>(src, dst_tensor, src_strides, dst_strides));
    } else {
      MMDEPLOY_ERROR("unsupported data type {}", src.data_type());
      return Status(eNotSupported);
    }
    dst = std::move(dst_tensor);
    return success();
  }

  template <typename T>
  Result<void> PermuteDispatch(const Tensor& src, Tensor& dst, const TensorStride& src_strides,
                               const TensorStride& dst_strides) {
    auto src_data = src.data<T>();
    auto dst_data = dst.data<T>();
    auto ndim = src.shape().size();
    auto total = src.size();
    impl::Permute(src_data, src_strides, dst_data, dst_strides, ndim, total,
                  GetNative<cudaStream_t>(stream()));
    return success();
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Permute, (cuda, 0),
                               []() { return std::make_unique<PermuteImpl>(); });

}  // namespace mmdeploy::operation::cuda
