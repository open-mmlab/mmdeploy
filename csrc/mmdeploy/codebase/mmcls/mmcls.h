// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODEBASE_MMCLS_MMCLS_H_
#define MMDEPLOY_SRC_CODEBASE_MMCLS_MMCLS_H_

#include "mmdeploy/codebase/common.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/serialization.h"

namespace mmdeploy::mmcls {

struct Label {
  int label_id;
  float score;
  MMDEPLOY_ARCHIVE_MEMBERS(label_id, score);
};

using Labels = std::vector<Label>;

MMDEPLOY_DECLARE_CODEBASE(MMClassification, mmcls);

}  // namespace mmdeploy::mmcls

#endif  // MMDEPLOY_SRC_CODEBASE_MMCLS_MMCLS_H_
