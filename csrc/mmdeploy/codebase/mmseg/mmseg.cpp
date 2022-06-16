// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmseg/mmseg.h"

using namespace std;

namespace mmdeploy {
namespace mmseg {

REGISTER_CODEBASE(MMSegmentation);

}

MMDEPLOY_DEFINE_REGISTRY(mmseg::MMSegmentation);
}  // namespace mmdeploy
