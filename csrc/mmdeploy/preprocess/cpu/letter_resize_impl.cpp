// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/letter_resize.h"
#include "opencv_utils.h"

using namespace std;

namespace mmdeploy {
namespace cpu {

class LetterResizeImpl final : public ::mmdeploy::LetterResizeImpl {
 public:
  LetterResizeImpl(const Value& args) : ::mmdeploy::LetterResizeImpl(args) {}
  ~LetterResizeImpl() = default;

 protected:
  Result<Tensor> ResizeImage(const Tensor& img, int dst_h, int dst_w) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(img, device_, stream_));

    SyncOnScopeExit(stream_, src_tensor.buffer() != img.buffer(), src_tensor);

    auto src_mat = Tensor2CVMat(src_tensor);
    auto dst_mat = Resize(src_mat, dst_h, dst_w, arg_.interpolation);

    return CVMat2Tensor(dst_mat);
  }
  Result<Tensor> PadImage(const Tensor& img, const int& top, const int& left, const int& bottom,
                          const int& right, const float& pad_val) override {
    OUTCOME_TRY(auto tensor, MakeAvailableOnDevice(img, device_, stream_));

    SyncOnScopeExit(stream_, tensor.buffer() != img.buffer(), tensor);

    cv::Mat dst_mat =
        Pad(Tensor2CVMat(tensor), top, left, bottom, right, cv::BORDER_CONSTANT, pad_val);
    return CVMat2Tensor(dst_mat);
  }
};

class LetterResizeImplCreator : public Creator<mmdeploy::LetterResizeImpl> {
 public:
  const char* GetName() const override { return "cpu"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override { return std::make_unique<LetterResizeImpl>(args); }
};

}  // namespace cpu
}  // namespace mmdeploy

using mmdeploy::LetterResizeImpl;
using mmdeploy::cpu::LetterResizeImplCreator;
REGISTER_MODULE(LetterResizeImpl, LetterResizeImplCreator);
