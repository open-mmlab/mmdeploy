// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/resize.h"

using namespace std;

namespace mmdeploy {
namespace elena {

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

class ResizeImplCreator : public Creator<mmdeploy::ResizeImpl> {
 public:
  const char* GetName() const override { return "elena"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override { return std::make_unique<ResizeImpl>(args); }
};

}  // namespace elena
}  // namespace mmdeploy

using mmdeploy::ResizeImpl;
using mmdeploy::elena::ResizeImplCreator;
REGISTER_MODULE(ResizeImpl, ResizeImplCreator);
