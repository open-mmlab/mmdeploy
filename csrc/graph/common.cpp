// Copyright (c) OpenMMLab. All rights reserved.

#include "common.h"

#include "archive/value_archive.h"

mmdeploy::graph::BaseNode::BaseNode(const mmdeploy::Value& cfg) {
  from_value(cfg["input"], inputs_);
  from_value(cfg["output"], outputs_);
  name_ = cfg.value<std::string>("name", "");
}
