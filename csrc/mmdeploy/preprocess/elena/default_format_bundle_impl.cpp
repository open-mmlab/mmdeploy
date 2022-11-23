// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/default_format_bundle.h"

namespace mmdeploy::elena {

class DefaultFormatBundleImpl : public ::mmdeploy::DefaultFormatBundleImpl {
 public:
  explicit DefaultFormatBundleImpl(const Value& args) : ::mmdeploy::DefaultFormatBundleImpl(args) {}

 protected:
  Result<Tensor> ToFloat32(const Tensor& tensor, const bool& img_to_float) override {
    auto& src_desc = tensor.desc();
    auto data_type = src_desc.data_type;
    auto shape = src_desc.shape;

    if (img_to_float && data_type == DataType::kINT8) {
      data_type = DataType::kFLOAT;
    }

    TensorDesc dummy_desc = {Device{"cpu"}, data_type, shape};
    Tensor dummy(dummy_desc, dummy_buffer_);

    return dummy;
  }

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

MMDEPLOY_REGISTER_TRANSFORM_IMPL(::mmdeploy::DefaultFormatBundleImpl, (elena, 0),
                                 DefaultFormatBundleImpl);

}  // namespace mmdeploy::elena
