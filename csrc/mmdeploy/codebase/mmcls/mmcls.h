// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODEBASE_MMCLS_MMCLS_H_
#define MMDEPLOY_SRC_CODEBASE_MMCLS_MMCLS_H_

#include "mmdeploy/codebase/common.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/serialization.h"

namespace mmdeploy {
namespace mmcls {

struct ClassifyOutput {
  struct Label {
    int label_id;
    float score;
    MMDEPLOY_ARCHIVE_MEMBERS(label_id, score);
  };
  std::vector<Label> labels;
  MMDEPLOY_ARCHIVE_MEMBERS(labels);
};

DECLARE_CODEBASE(MMClassification, mmcls);
}  // namespace mmcls

MMDEPLOY_DECLARE_REGISTRY(mmcls::MMClassification);
}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CODEBASE_MMCLS_MMCLS_H_
