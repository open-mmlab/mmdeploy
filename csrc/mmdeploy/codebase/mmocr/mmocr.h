// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_MMOCR_H
#define MMDEPLOY_MMOCR_H

#include <array>

#include "mmdeploy/codebase/common.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/module.h"

namespace mmdeploy::mmocr {

struct TextDetection {
  std::array<float, 8> bbox;
  float score;
  MMDEPLOY_ARCHIVE_MEMBERS(bbox, score);
};

using TextDetections = std::vector<TextDetection>;

struct TextRecognition {
  std::string text;
  std::vector<float> score;
  MMDEPLOY_ARCHIVE_MEMBERS(text, score);
};

MMDEPLOY_DECLARE_CODEBASE(MMOCR, mmocr);

}  // namespace mmdeploy::mmocr

#endif  // MMDEPLOY_MMOCR_H
