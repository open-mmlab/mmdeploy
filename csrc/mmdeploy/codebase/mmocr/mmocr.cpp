// Copyright (c) OpenMMLab. All rights reserved.

#include "codebase/mmocr/mmocr.h"

#include "core/registry.h"
#include "core/utils/formatter.h"

namespace mmdeploy {
namespace mmocr {

REGISTER_CODEBASE(MMOCR);

}  // namespace mmocr

MMDEPLOY_DEFINE_REGISTRY(mmocr::MMOCR);
}  // namespace mmdeploy
