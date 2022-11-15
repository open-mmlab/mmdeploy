// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmocr/mmocr.h"

#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy {
namespace mmocr {

REGISTER_CODEBASE(MMOCR);

}  // namespace mmocr

MMDEPLOY_DEFINE_REGISTRY(mmocr::MMOCR);
}  // namespace mmdeploy
