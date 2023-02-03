// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_MMSEG_H
#define MMDEPLOY_MMSEG_H

#include "mmdeploy/codebase/common.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/tensor.h"

namespace mmdeploy::mmseg {

struct SegmentorOutput {
  Tensor mask;
  Tensor score;
  int height;
  int width;
  int classes;
  MMDEPLOY_ARCHIVE_MEMBERS(mask, score, height, width, classes);
};

MMDEPLOY_DECLARE_CODEBASE(MMSegmentation, mmseg);

}  // namespace mmdeploy::mmseg

#endif  // MMDEPLOY_MMSEG_H
