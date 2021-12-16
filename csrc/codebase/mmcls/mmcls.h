// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODEBASE_MMCLS_MMCLS_H_
#define MMDEPLOY_SRC_CODEBASE_MMCLS_MMCLS_H_

#include "codebase/common.h"
#include "core/device.h"
#include "core/module.h"
#include "core/serialization.h"

namespace mmdeploy::mmcls {

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

}  // namespace mmdeploy::mmcls

#endif  // MMDEPLOY_SRC_CODEBASE_MMCLS_MMCLS_H_
