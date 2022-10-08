// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/normalize.h"

using namespace std;

namespace mmdeploy {
namespace elena {

class NormalizeImpl : public ::mmdeploy::NormalizeImpl {
 public:
  NormalizeImpl(const Value& value) : ::mmdeploy::NormalizeImpl(value){};
  ~NormalizeImpl() = default;

 protected:
  Result<Tensor> NormalizeImage(const Tensor& tensor) override {
    auto& src_desc = tensor.desc();
    auto data_type = DataType::kFLOAT;
    auto shape = src_desc.shape;

    TensorDesc dummy_desc = {Device{"cpu"}, data_type, shape};
    Tensor dummy(dummy_desc, dummy_buffer_);

    return dummy;
  }
  Buffer dummy_buffer_{Device{"cpu"}, 0, nullptr};
};

class NormalizeImplCreator : public Creator<::mmdeploy::NormalizeImpl> {
 public:
  const char* GetName() const override { return "elena"; }
  int GetVersion() const override { return 1; }
  std::unique_ptr<::mmdeploy::NormalizeImpl> Create(const Value& args) override {
    return make_unique<NormalizeImpl>(args);
  }
};

}  // namespace elena
}  // namespace mmdeploy

using mmdeploy::NormalizeImpl;
using mmdeploy::elena::NormalizeImplCreator;
REGISTER_MODULE(NormalizeImpl, NormalizeImplCreator);
