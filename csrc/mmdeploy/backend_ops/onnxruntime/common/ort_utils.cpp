// Copyright (c) OpenMMLab. All rights reserved.
#include "ort_utils.h"

namespace mmdeploy {

CustomOpsTable& get_mmdeploy_custom_ops() {
  static CustomOpsTable _custom_ops;
  return _custom_ops;
}
}  // namespace mmdeploy
