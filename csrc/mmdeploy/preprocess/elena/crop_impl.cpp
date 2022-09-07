// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/crop.h"

using namespace std;

namespace mmdeploy {
namespace elena {

class CenterCropImpl : public ::mmdeploy::CenterCropImpl {
 public:
  explicit CenterCropImpl(const Value& args) : ::mmdeploy::CenterCropImpl(args) {}

 protected:
  Result<Tensor> CropImage(const Tensor& tensor, int top, int left, int bottom,
                           int right) override {
    auto& src_desc = tensor.desc();
    auto data_type = src_desc.data_type;
    auto shape = src_desc.shape;
    shape[1] = bottom - top + 1;  // h
    shape[2] = right - left + 1;  // w

    TensorDesc dummy_desc = {Device{"cpu"}, data_type, shape};
    Tensor dummy(dummy_desc, dummy_buffer_);

    return dummy;
  }
  Buffer dummy_buffer_{Device{"cpu"}, 0, nullptr};
};

class CenterCropImplCreator : public Creator<::mmdeploy::CenterCropImpl> {
 public:
  const char* GetName() const override { return "elena"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override { return make_unique<CenterCropImpl>(args); }
};

}  // namespace elena
}  // namespace mmdeploy

using ::mmdeploy::CenterCropImpl;
using ::mmdeploy::elena::CenterCropImplCreator;

REGISTER_MODULE(CenterCropImpl, CenterCropImplCreator);
