// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_CODEBASE_MMOCR_DBNET_H_
#define MMDEPLOY_CSRC_CODEBASE_MMOCR_DBNET_H_

#include "codebase/mmocr/mmocr.h"
#include "core/device.h"
#include "core/registry.h"
#include "core/serialization.h"
#include "core/value.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace mmdeploy {
namespace mmocr {

struct DbHeadParams {
  std::string text_repr_type{"quad"};
  float mask_thr{.3};
  float min_text_score{.3};
  int min_text_width{5};
  float unclip_ratio{1.5};
  int max_candidates{3000};
  bool rescale{true};
  float downsample_ratio{1.};
};

class DbHeadImpl {
 public:
  virtual void Init(const DbHeadParams& params, const Stream& stream) {
    params_ = &params;
    stream_ = stream;
  }

  virtual Result<void> Process(Tensor prob, std::vector<std::vector<cv::Point>>& points,
                               std::vector<float>& scores) = 0;

 protected:
  const DbHeadParams* params_{};
  Stream stream_;
};

}  // namespace mmocr

MMDEPLOY_DECLARE_REGISTRY(mmocr::DbHeadImpl);

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_CODEBASE_MMOCR_DBNET_H_
