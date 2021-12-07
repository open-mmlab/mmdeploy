// Copyright (c) OpenMMLab. All rights reserved.

#include "core/utils/formatter.h"

#include "archive/json_archive.h"
#include "core/value.h"

namespace mmdeploy {

std::string format_value(const Value& value) { return mmdeploy::to_json(value).dump(2); }

}  // namespace mmdeploy
