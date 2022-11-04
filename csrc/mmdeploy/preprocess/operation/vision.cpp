//
// Created by zhangli on 11/3/22.
//
#include "mmdeploy/preprocess/operation/vision.h"

namespace mmdeploy::operation {

MMDEPLOY_DEFINE_REGISTRY(ToBGR);
MMDEPLOY_DEFINE_REGISTRY(ToGray);
MMDEPLOY_DEFINE_REGISTRY(Resize);
MMDEPLOY_DEFINE_REGISTRY(Pad);
MMDEPLOY_DEFINE_REGISTRY(ToFloat);
MMDEPLOY_DEFINE_REGISTRY(HWC2CHW);
MMDEPLOY_DEFINE_REGISTRY(Normalize);
MMDEPLOY_DEFINE_REGISTRY(Crop);

}  // namespace mmdeploy::operation
