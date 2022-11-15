// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmedit/mmedit.h"

#include "mmdeploy/core/registry.h"

namespace mmdeploy {
namespace mmedit {

REGISTER_CODEBASE(MMEdit);

}  // namespace mmedit

MMDEPLOY_DEFINE_REGISTRY(mmedit::MMEdit);
}  // namespace mmdeploy
