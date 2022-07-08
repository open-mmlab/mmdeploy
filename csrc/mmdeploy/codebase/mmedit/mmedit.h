// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODEBASE_MMEDIT_MMEDIT_H_
#define MMDEPLOY_SRC_CODEBASE_MMEDIT_MMEDIT_H_

#include "mmdeploy/codebase/common.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/serialization.h"

namespace mmdeploy {
namespace mmedit {

using RestorerOutput = Mat;

DECLARE_CODEBASE(MMEdit, mmedit);

}  // namespace mmedit

MMDEPLOY_DECLARE_REGISTRY(mmedit::MMEdit);
}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CODEBASE_MMEDIT_MMEDIT_H_
