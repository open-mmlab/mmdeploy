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
  const Device& GetDevice() { return device(); }

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
