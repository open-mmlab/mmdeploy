// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/image2tensor.h"

namespace mmdeploy::elena {

class ImageToTensorImpl : public ::mmdeploy::ImageToTensorImpl {
 public:
  explicit ImageToTensorImpl(const Value& args) : ::mmdeploy::ImageToTensorImpl(args) {}

 protected:
  Result<Tensor> HWC2CHW(const Tensor& tensor) override {
    auto& src_desc = tensor.desc();
    auto data_type = src_desc.data_type;
    auto shape = src_desc.shape;
    shape = {shape[0], shape[3], shape[1], shape[2]};

    TensorDesc dummy_desc = {Device{"cpu"}, data_type, shape};
    Tensor dummy(dummy_desc, dummy_buffer_);

    return dummy;
  }
  Buffer dummy_buffer_{Device{"cpu"}, 0, nullptr};
};

MMDEPLOY_REGISTER_TRANSFORM_IMPL(::mmdeploy::ImageToTensorImpl, (elena, 0), ImageToTensorImpl);

}  // namespace mmdeploy::elena
