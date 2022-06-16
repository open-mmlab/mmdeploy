// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_CODEBASE_MMOCR_PANET_H_
#define MMDEPLOY_CSRC_CODEBASE_MMOCR_PANET_H_

#include "mmdeploy/codebase/mmocr/mmocr.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "opencv2/core.hpp"

namespace mmdeploy {
namespace mmocr {

class PaHeadImpl {
 public:
  virtual ~PaHeadImpl() = default;

  virtual void Init(const Stream& stream) { stream_ = stream; }

  virtual Result<void> Process(Tensor text_pred,             //
                               Tensor kernel_pred,           //
                               Tensor embed_pred,            //
                               float min_text_confidence,    //
                               float min_kernel_confidence,  //
                               cv::Mat_<float>& text_score,  //
                               cv::Mat_<uint8_t>& text,      //
                               cv::Mat_<uint8_t>& kernel,    //
                               cv::Mat_<int>& label,         //
                               cv::Mat_<float>& embed,       //
                               int& region_num) = 0;

 protected:
  Stream stream_;
};

}  // namespace mmocr

MMDEPLOY_DECLARE_REGISTRY(mmocr::PaHeadImpl);

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_CODEBASE_MMOCR_PANET_H_
