// Copyright (c) OpenMMLab. All rights reserved.

#include "common.h"

#include "mmdeploy/archive/value_archive.h"

mmdeploy::graph::BaseNode::BaseNode(const mmdeploy::Value& cfg) {
  try {
    from_value(cfg["input"], inputs_);
    from_value(cfg["output"], outputs_);
    name_ = cfg.value<std::string>("name", "");
  } catch (...) {
    MMDEPLOY_ERROR("error parsing config: {}", cfg);
    throw;
  }
}
