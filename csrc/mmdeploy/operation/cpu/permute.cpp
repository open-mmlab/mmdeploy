// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class PermuteImpl : public Permute {
 public:
  explicit PermuteImpl() {}

  Result<void> apply(const Tensor& src, Tensor& dst, const std::vector<int>& axes) override {
    int ndim = src.shape().size();
    if (ndim != axes.size()) {
      MMDEPLOY_ERROR("The size of axes should be equal to src, {} vs {}", axes.size(), ndim);
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

    std::vector<int> dst_strides(ndim);
    std::vector<int> src_strides(ndim);
    dst_strides[ndim - 1] = src_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
      dst_strides[i] = dst_strides[i + 1] * dst_dims[i + 1];
      src_strides[i] = src_strides[i + 1] * src_dims[i + 1];
    }

    std::vector<int> tmp(ndim);
    for (int i = 0; i < ndim; i++) {
      tmp[i] = src_strides[axes[i]];
    }
    src_strides.swap(tmp);

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
  Result<void> PermuteDispatch(const Tensor& src, Tensor& dst, const std::vector<int>& src_strides,
                               const std::vector<int>& dst_strides) {
    auto shape = dst.shape();
    int ndim = src.shape().size();
    std::vector<int> coord(ndim, 0);
    auto dst_data = dst.data<T>();
    auto src_data = src.data<T>();

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
    return success();
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Permute, (cpu, 0), []() { return std::make_unique<PermuteImpl>(); });

}  // namespace mmdeploy::operation::cpu
