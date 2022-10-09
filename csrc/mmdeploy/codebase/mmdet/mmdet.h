// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODEBASE_MMDET_MMDET_H_
#define MMDEPLOY_SRC_CODEBASE_MMDET_MMDET_H_

#include <array>

#include "mmdeploy/codebase/common.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/serialization.h"

namespace mmdeploy {
namespace mmdet {

struct Detection {
  int index;
  int label_id;
  float score;
  std::array<float, 4> bbox;  // left, top, right, bottom
  Mat mask;
  MMDEPLOY_ARCHIVE_MEMBERS(index, label_id, score, bbox, mask);
};

using Detections = std::vector<Detection>;

DECLARE_CODEBASE(MMDetection, mmdet);
}  // namespace mmdet

MMDEPLOY_DECLARE_REGISTRY(mmdet::MMDetection);
}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CODEBASE_MMDET_MMDET_H_
