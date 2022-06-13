// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/formatter.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/value.h"

namespace mmdeploy {

std::string format_value(const Value& value) { return mmdeploy::to_json(value).dump(2); }

}  // namespace mmdeploy
