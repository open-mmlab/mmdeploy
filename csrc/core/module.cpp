// Copyright (c) OpenMMLab. All rights reserved.

#include "module.h"

#include "registry.h"

namespace mmdeploy {

template class Registry<Module>;
template class Creator<Module>;

}  // namespace mmdeploy
