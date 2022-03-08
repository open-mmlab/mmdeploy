// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODEBASE_MMDET_MMDET_H_
#define MMDEPLOY_SRC_CODEBASE_MMDET_MMDET_H_

#include <array>

#include "codebase/common.h"
#include "core/device.h"
#include "core/mat.h"
#include "core/module.h"
#include "core/registry.h"
#include "core/serialization.h"

namespace mmdeploy {
namespace mmdet {

struct DetectorOutput {
  struct Detection {
    int index;
    int label_id;
    float score;
    std::array<float, 4> bbox;  // left, top, right, bottom
    Mat mask;
    MMDEPLOY_ARCHIVE_MEMBERS(index, label_id, score, bbox, mask);
  };
  std::vector<Detection> detections;
  MMDEPLOY_ARCHIVE_MEMBERS(detections);
};

DECLARE_CODEBASE(MMDetection, mmdet);
}  // namespace mmdet

MMDEPLOY_DECLARE_REGISTRY(mmdet::MMDetection);
}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CODEBASE_MMDET_MMDET_H_
