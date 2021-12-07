// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODEBASE_MMDET_MMDET_H_
#define MMDEPLOY_SRC_CODEBASE_MMDET_MMDET_H_

#include "codebase/common.h"
#include "core/device.h"
#include "core/module.h"
#include "core/serialization.h"

namespace mmdeploy::mmdet {

struct DetectorOutput {
  struct Detection {
    int label_id;
    float score;
    std::array<float, 4> bbox;  // left, top, right, bottom
    MMDEPLOY_ARCHIVE_MEMBERS(label_id, score, bbox);
  };
  std::vector<Detection> detections;
  MMDEPLOY_ARCHIVE_MEMBERS(detections);
};

DECLARE_CODEBASE(MMDetPostprocess);

}  // namespace mmdeploy::mmdet

#endif  // MMDEPLOY_SRC_CODEBASE_MMDET_MMDET_H_
