// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmocr/psenet.h"

#include "mmdeploy/core/utils/device_utils.h"
#include "opencv2/imgproc.hpp"

namespace mmdeploy::mmocr {

class PseHeadCpuImpl : public PseHeadImpl {
 public:
  PseHeadCpuImpl() : device_(0) {}

  Result<void> Process(Tensor preds,                 //
                       float min_kernel_confidence,  //
                       cv::Mat_<float>& score,       //
                       cv::Mat_<uint8_t>& masks,     //
                       cv::Mat_<int>& label,         //
                       int& region_num) override {
    OUTCOME_TRY(preds, MakeAvailableOnDevice(preds, device_, stream_));
    OUTCOME_TRY(stream_.Wait());

    auto channels = static_cast<int>(preds.shape(0));
    auto height = static_cast<int>(preds.shape(1));
    auto width = static_cast<int>(preds.shape(2));

    cv::Mat_<float> probs(channels, height * width, preds.data<float>());
    sigmoid(probs);

    probs.row(0).reshape(1, height).copyTo(score);

    masks = probs > min_kernel_confidence;

    for (int i = 1; i < channels; ++i) {
      masks.row(i) &= masks.row(0);
    }

    cv::Mat_<uint8_t> kernel = masks.row(channels - 1).reshape(1, height);
    region_num = cv::connectedComponents(kernel, label, 4, CV_32S);

    return success();
  }

  static void sigmoid(cv::Mat_<float>& score) {
    cv::exp(-score, score);
    score = 1 / (1 + score);
  }

 private:
  Device device_;
};

class PseHeadCpuImplCreator : public ::mmdeploy::Creator<PseHeadImpl> {
 public:
  const char* GetName() const override { return "cpu"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<PseHeadImpl> Create(const Value&) override {
    return std::make_unique<PseHeadCpuImpl>();
  }
};

REGISTER_MODULE(PseHeadImpl, PseHeadCpuImplCreator);

}  // namespace mmdeploy::mmocr
