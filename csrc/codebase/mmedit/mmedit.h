// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODEBASE_MMEDIT_MMEDIT_H_
#define MMDEPLOY_SRC_CODEBASE_MMEDIT_MMEDIT_H_

#include "codebase/common.h"
#include "core/device.h"
#include "core/mat.h"
#include "core/module.h"
#include "core/serialization.h"

namespace mmdeploy::mmedit {

using RestorerOutput = Mat;

DECLARE_CODEBASE(MMEdit, mmedit);

}  // namespace mmdeploy::mmedit

#endif  // MMDEPLOY_SRC_CODEBASE_MMEDIT_MMEDIT_H_
