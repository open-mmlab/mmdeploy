// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_MMPOSE_H
#define MMDEPLOY_MMPOSE_H

#include <array>

#include "mmdeploy/codebase/common.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/module.h"

namespace mmdeploy::mmpose {

struct PoseDetectorOutput {
  struct KeyPoint {
    std::array<float, 2> bbox;  // x, y
    float score;
    MMDEPLOY_ARCHIVE_MEMBERS(bbox, score);
  };
  std::vector<KeyPoint> key_points;
  MMDEPLOY_ARCHIVE_MEMBERS(key_points);
};

MMDEPLOY_DECLARE_CODEBASE(MMPose, mmpose);

}  // namespace mmdeploy::mmpose

#endif  // MMDEPLOY_MMPOSE_H
