// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/pad.h"
#include "opencv_utils.h"

using namespace std;

namespace mmdeploy {
namespace cpu {

class PadImpl : public ::mmdeploy::PadImpl {
 public:
  PadImpl(const Value& args) : ::mmdeploy::PadImpl(args) {
    static map<string, int> border_map{{"constant", cv::BORDER_CONSTANT},
                                       {"edge", cv::BORDER_REPLICATE},
                                       {"reflect", cv::BORDER_REFLECT_101},
                                       {"symmetric", cv::BORDER_REFLECT}};
    if (border_map.find(arg_.padding_mode) == border_map.end()) {
      MMDEPLOY_ERROR("unsupported padding_mode '{}'", arg_.padding_mode);
      throw std::invalid_argument("unsupported padding_mode");
    }
    border_type_ = border_map[arg_.padding_mode];
  }

 protected:
  Result<Tensor> PadImage(const Tensor& img, const std::array<int, 4>& padding) override {
    OUTCOME_TRY(auto tensor, MakeAvailableOnDevice(img, device_, stream_));

    SyncOnScopeExit(stream_, tensor.buffer() != img.buffer(), tensor);

    cv::Mat dst_mat = Pad(Tensor2CVMat(tensor), padding[1], padding[0], padding[3], padding[2],
                          border_type_, arg_.pad_val);
    return CVMat2Tensor(dst_mat);
  }

 private:
  int border_type_;
};

class PadImplCreator : public Creator<::mmdeploy::PadImpl> {
 public:
  const char* GetName() const override { return "cpu"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override { return make_unique<PadImpl>(args); }
};

}  // namespace cpu
}  // namespace mmdeploy

using mmdeploy::PadImpl;
using mmdeploy::cpu::PadImplCreator;
REGISTER_MODULE(PadImpl, PadImplCreator);
