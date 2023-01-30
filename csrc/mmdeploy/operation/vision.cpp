// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/vision.h"

namespace mmdeploy::operation {

MMDEPLOY_DEFINE_REGISTRY(CvtColor);
MMDEPLOY_DEFINE_REGISTRY(Resize);
MMDEPLOY_DEFINE_REGISTRY(Pad);
MMDEPLOY_DEFINE_REGISTRY(ToFloat);
MMDEPLOY_DEFINE_REGISTRY(HWC2CHW);
MMDEPLOY_DEFINE_REGISTRY(Normalize);
MMDEPLOY_DEFINE_REGISTRY(Crop);
MMDEPLOY_DEFINE_REGISTRY(Flip);
MMDEPLOY_DEFINE_REGISTRY(WarpAffine);
MMDEPLOY_DEFINE_REGISTRY(CropResizePad);

}  // namespace mmdeploy::operation
