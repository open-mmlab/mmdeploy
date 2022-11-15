// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_MMROTATE_H
#define MMDEPLOY_MMROTATE_H

#include <array>

#include "mmdeploy/codebase/common.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/module.h"

namespace mmdeploy {
namespace mmrotate {

struct RotatedDetectorOutput {
  struct Detection {
    int label_id;
    float score;
    std::array<float, 5> rbbox;  // cx,cy,w,h,ag
    MMDEPLOY_ARCHIVE_MEMBERS(label_id, score, rbbox);
  };
  std::vector<Detection> detections;
  MMDEPLOY_ARCHIVE_MEMBERS(detections);
};

DECLARE_CODEBASE(MMRotate, mmrotate);
}  // namespace mmrotate

MMDEPLOY_DECLARE_REGISTRY(mmrotate::MMRotate);
}  // namespace mmdeploy

#endif  // MMDEPLOY_MMROTATE_H
