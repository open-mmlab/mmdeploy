// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/pad.h"

using namespace std;

namespace mmdeploy {
namespace elena {

class PadImpl : public ::mmdeploy::PadImpl {
 public:
  PadImpl(const Value& args) : ::mmdeploy::PadImpl(args) {}

 protected:
  Result<Tensor> PadImage(const Tensor& img, const std::array<int, 4>& padding) override {
    auto& src_desc = img.desc();
    auto data_type = src_desc.data_type;
    auto shape = src_desc.shape;  // 1 x h x w x c
    shape[1] += padding[1] + padding[3];
    shape[2] += padding[0] + padding[2];

    TensorDesc dummy_desc = {Device{"cpu"}, data_type, shape};
    Tensor dummy(dummy_desc, dummy_buffer_);

    return dummy;
  }
  Buffer dummy_buffer_{Device{"cpu"}, 0, nullptr};
};

class PadImplCreator : public Creator<::mmdeploy::PadImpl> {
 public:
  const char* GetName() const override { return "elena"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override { return make_unique<PadImpl>(args); }
};

}  // namespace elena
}  // namespace mmdeploy

using mmdeploy::PadImpl;
using mmdeploy::elena::PadImplCreator;
REGISTER_MODULE(PadImpl, PadImplCreator);
