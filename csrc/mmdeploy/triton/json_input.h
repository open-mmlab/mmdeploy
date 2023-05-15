// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_TRITON_JSON_INPUT_H
#define MMDEPLOY_TRITON_JSON_INPUT_H

#include <array>
#include <string>

#include "mmdeploy/archive/value_archive.h"

namespace triton::backend::mmdeploy {

struct TextBbox {
  std::array<float, 8> bbox;
  MMDEPLOY_ARCHIVE_MEMBERS(bbox);
};

void CreateJsonInput(::mmdeploy::Value &input, const std::string &type, ::mmdeploy::Value &output);

}  // namespace triton::backend::mmdeploy

#endif  // MMDEPLOY_TRITON_JSON_INPUT_H
