// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_CODEBASE_MMOCR_DBNET_H_
#define MMDEPLOY_CSRC_CODEBASE_MMOCR_DBNET_H_

#include "mmdeploy/codebase/mmocr/mmocr.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "opencv2/core.hpp"

namespace mmdeploy::mmocr {

class DbHeadImpl {
 public:
  virtual ~DbHeadImpl() = default;

  virtual void Init(const Stream& stream) { stream_ = stream; }

  virtual Result<void> Process(Tensor prob, float mask_thr, int max_candidates,
                               std::vector<std::vector<cv::Point>>& points,
                               std::vector<float>& scores) = 0;

 protected:
  Stream stream_;
};

MMDEPLOY_DECLARE_REGISTRY(DbHeadImpl, std::unique_ptr<DbHeadImpl>());

}  // namespace mmdeploy::mmocr

#endif  // MMDEPLOY_CSRC_CODEBASE_MMOCR_DBNET_H_
