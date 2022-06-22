// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/graph/common.h"

#include "mmdeploy/archive/value_archive.h"

namespace mmdeploy::graph {

Result<std::vector<std::string>> ParseStringArray(const Value& value) {
  if (value.is_string()) {
    return std::vector{value.get<std::string>()};
  } else if (value.is_array()) {
    return from_value<std::vector<std::string>>(value);
  }
  return Status(eInvalidArgument);
}

}  // namespace mmdeploy::graph
