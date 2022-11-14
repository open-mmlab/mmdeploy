// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/normalize.h"
#include "opencv_utils.h"

using namespace std;

namespace mmdeploy {
namespace cpu {

class NormalizeImpl : public ::mmdeploy::NormalizeImpl {
 public:
  NormalizeImpl(const Value& value) : ::mmdeploy::NormalizeImpl(value){};
  ~NormalizeImpl() = default;

 protected:
  Result<Tensor> NormalizeImage(const Tensor& tensor) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device_, stream_));

    SyncOnScopeExit(stream_, src_tensor.buffer() != tensor.buffer(), src_tensor);

    auto mat = Tensor2CVMat(src_tensor);
    auto dst_mat = Normalize(mat, arg_.mean, arg_.std, arg_.to_rgb, true);
    return CVMat2Tensor(dst_mat);
  }
};

class NormalizeImplCreator : public Creator<::mmdeploy::NormalizeImpl> {
 public:
  const char* GetName() const override { return "cpu"; }
  int GetVersion() const override { return 1; }
  std::unique_ptr<::mmdeploy::NormalizeImpl> Create(const Value& args) override {
    return make_unique<NormalizeImpl>(args);
  }
};

}  // namespace cpu
}  // namespace mmdeploy

using mmdeploy::NormalizeImpl;
using mmdeploy::cpu::NormalizeImplCreator;
REGISTER_MODULE(NormalizeImpl, NormalizeImplCreator);
