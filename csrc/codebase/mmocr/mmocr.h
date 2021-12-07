// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_MMOCR_H
#define MMDEPLOY_MMOCR_H

#include "codebase/common.h"
#include "core/device.h"
#include "core/module.h"

namespace mmdeploy::mmocr {

struct TextDetectorOutput {
  std::vector<std::array<float, 8>> boxes;
  std::vector<float> scores;
  MMDEPLOY_ARCHIVE_MEMBERS(boxes, scores);
};

struct TextRecognizerOutput {
  std::string text;
  std::vector<float> score;
  MMDEPLOY_ARCHIVE_MEMBERS(text, score);
};

DECLARE_CODEBASE(MMOCRPostprocess);

}  // namespace mmdeploy::mmocr

#endif  // MMDEPLOY_MMOCR_H
