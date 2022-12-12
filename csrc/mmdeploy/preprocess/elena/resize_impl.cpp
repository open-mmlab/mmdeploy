// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/resize.h"

using namespace std;

namespace mmdeploy::elena {

class ResizeImpl final : public ::mmdeploy::ResizeImpl {
 public:
  ResizeImpl(const Value& args) : ::mmdeploy::ResizeImpl(args) {}
  ~ResizeImpl() = default;

 protected:
  Result<Tensor> ResizeImage(const Tensor& img, int dst_h, int dst_w) override {
    auto& src_desc = img.desc();
    auto data_type = src_desc.data_type;
    TensorShape shape = {1, dst_h, dst_w, img.shape().back()};

    TensorDesc dummy_desc = {Device{"cpu"}, data_type, shape};
    Tensor dummy(dummy_desc, dummy_buffer_);

    return dummy;
  }
  Buffer dummy_buffer_{Device{"cpu"}, 0, nullptr};
};

MMDEPLOY_REGISTER_TRANSFORM_IMPL(::mmdeploy::ResizeImpl, (elena, 0), ResizeImpl);

}  // namespace mmdeploy::elena
