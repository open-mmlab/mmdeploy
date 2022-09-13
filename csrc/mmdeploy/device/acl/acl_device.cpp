// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/device_impl.h"

namespace mmdeploy::framework {

class AclPlatformRegisterer {
 public:
  AclPlatformRegisterer() { gPlatformRegistry().AddAlias("npu", "cpu"); }
};

AclPlatformRegisterer g_acl_platform_registerer;

}  // namespace mmdeploy::framework
