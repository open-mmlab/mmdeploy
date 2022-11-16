// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_CODEBASE_MMOCR_PSENET_H_
#define MMDEPLOY_CSRC_CODEBASE_MMOCR_PSENET_H_

#include "mmdeploy/codebase/mmocr/mmocr.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "opencv2/core.hpp"

namespace mmdeploy::mmocr {

class PseHeadImpl {
 public:
  virtual ~PseHeadImpl() = default;

  virtual void Init(const Stream& stream) { stream_ = stream; }

  virtual Result<void> Process(Tensor preds,                 //
                               float min_kernel_confidence,  //
                               cv::Mat_<float>& score,       //
                               cv::Mat_<uint8_t>& masks,     //
                               cv::Mat_<int>& label,         //
                               int& region_num) = 0;

 protected:
  Stream stream_;
};

MMDEPLOY_DECLARE_REGISTRY(PseHeadImpl, std::unique_ptr<PseHeadImpl>());

}  // namespace mmdeploy::mmocr

#endif  // MMDEPLOY_CSRC_CODEBASE_MMOCR_PSENET_H_
