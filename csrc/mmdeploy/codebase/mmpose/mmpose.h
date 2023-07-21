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
  struct BBox {
    std::array<float, 4> boundingbox;  // x1,y1,x2,y2
    float score;
    MMDEPLOY_ARCHIVE_MEMBERS(boundingbox, score);
  };
  std::vector<KeyPoint> key_points;
  std::vector<BBox> detections;
  MMDEPLOY_ARCHIVE_MEMBERS(key_points, detections);
};


MMDEPLOY_DECLARE_CODEBASE(MMPose, mmpose);

}  // namespace mmdeploy::mmpose

#endif  // MMDEPLOY_MMPOSE_H
