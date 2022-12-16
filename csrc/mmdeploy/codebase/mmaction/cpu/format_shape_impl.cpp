// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmaction/format_shape.h"
#include "mmdeploy/core/utils/device_utils.h"

using namespace std;

namespace mmdeploy::mmaction::cpu {

class FormatShapeImpl : public FormatShapeOp {
 public:
  explicit FormatShapeImpl(std::string input_format) : FormatShapeOp(std::move(input_format)) {}

 protected:
  Device host_{0, 0};

  const Device& GetDevice() { return host_; }

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
};

MMDEPLOY_REGISTER_FACTORY_FUNC(FormatShapeOp, (cpu, 0), [](std::string input_format) {
  return std::make_unique<FormatShapeImpl>(std::move(input_format));
});

}  // namespace mmdeploy::mmaction::cpu
