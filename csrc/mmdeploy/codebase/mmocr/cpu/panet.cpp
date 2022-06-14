// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmocr/panet.h"

#include "opencv2/imgproc.hpp"

namespace mmdeploy::mmocr {

class PaHeadCpuImpl : public PaHeadImpl {
 public:
  Result<void> Process(Tensor text_pred,             //
                       Tensor kernel_pred,           //
                       Tensor embed_pred,            //
                       float min_text_confidence,    //
                       float min_kernel_confidence,  //
                       cv::Mat_<float>& text_score,  //
                       cv::Mat_<uint8_t>& text,      //
                       cv::Mat_<uint8_t>& kernel,    //
                       cv::Mat_<int>& label,         //
                       cv::Mat_<float>& embed,       //
                       int& region_num) override {
    OUTCOME_TRY(stream_.Wait());

    auto height = static_cast<int>(text_pred.shape(1));
    auto width = static_cast<int>(text_pred.shape(2));

    text_score = cv::Mat_<float>(height, width, text_pred.data<float>());
    sigmoid(text_score);

    text = text_score > min_text_confidence;

    cv::Mat_<float> kernel_score(height, width, kernel_pred.data<float>());
    sigmoid(kernel_score);

    kernel = kernel_score > min_kernel_confidence & text;

    auto n_embed_channels = static_cast<int>(embed_pred.shape(0));
    embed = cv::Mat_<float>(n_embed_channels, height * width, embed_pred.data<float>());
    cv::transpose(embed, embed);

    region_num = cv::connectedComponents(kernel, label, 4, CV_32S);

    return success();
  }

  static void sigmoid(cv::Mat_<float>& score) {
    cv::exp(-score, score);
    score = 1 / (1 + score);
  }
};

class PaHeadCpuImplCreator : public ::mmdeploy::Creator<PaHeadImpl> {
 public:
  const char* GetName() const override { return "cpu"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<PaHeadImpl> Create(const Value&) override {
    return std::make_unique<PaHeadCpuImpl>();
  }
};

REGISTER_MODULE(PaHeadImpl, PaHeadCpuImplCreator);

}  // namespace mmdeploy::mmocr
