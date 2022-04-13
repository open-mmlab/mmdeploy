// Copyright (c) OpenMMLab. All rights reserved.

#include "core/registry.h"
#include "core/tensor.h"
#include "core/utils/device_utils.h"
#include "opencv_utils.h"
#include "preprocess/transform/resize.h"

using namespace std;

namespace mmdeploy {
namespace cpu {

class ResizeImpl final : public ::mmdeploy::ResizeImpl {
 public:
  ResizeImpl(const Value& args) : ::mmdeploy::ResizeImpl(args) {}
  ~ResizeImpl() = default;

 protected:
  Result<Tensor> ResizeImage(const Tensor& img, int dst_h, int dst_w) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(img, device_, stream_));

    SyncOnScopeExit(stream_, src_tensor.buffer() != img.buffer(), src_tensor);

    auto src_mat = Tensor2CVMat(src_tensor);
    auto dst_mat = Resize(src_mat, dst_h, dst_w, arg_.interpolation);

    return CVMat2Tensor(dst_mat);
  }
};

class ResizeImplCreator : public Creator<mmdeploy::ResizeImpl> {
 public:
  const char* GetName() const override { return "cpu"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override { return std::make_unique<ResizeImpl>(args); }
};

}  // namespace cpu
}  // namespace mmdeploy

using mmdeploy::ResizeImpl;
using mmdeploy::cpu::ResizeImplCreator;
REGISTER_MODULE(ResizeImpl, ResizeImplCreator);
