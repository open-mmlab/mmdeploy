// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODEBASE_MMEDIT_MMEDIT_H_
#define MMDEPLOY_SRC_CODEBASE_MMEDIT_MMEDIT_H_

#include "codebase/common.h"
#include "core/device.h"
#include "core/mat.h"
#include "core/module.h"
#include "core/serialization.h"

namespace mmdeploy {
namespace mmedit {

using RestorerOutput = Mat;

DECLARE_CODEBASE(MMEdit, mmedit);

}  // namespace mmedit

MMDEPLOY_DECLARE_REGISTRY(mmedit::MMEdit);
}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CODEBASE_MMEDIT_MMEDIT_H_
