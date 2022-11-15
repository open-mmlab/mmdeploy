// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODEBASE_MMEDIT_MMEDIT_H_
#define MMDEPLOY_SRC_CODEBASE_MMEDIT_MMEDIT_H_

#include "mmdeploy/codebase/common.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/serialization.h"

namespace mmdeploy::mmedit {

using RestorerOutput = Mat;

MMDEPLOY_DECLARE_CODEBASE(MMEdit, mmedit);

}  // namespace mmdeploy::mmedit

#endif  // MMDEPLOY_SRC_CODEBASE_MMEDIT_MMEDIT_H_
