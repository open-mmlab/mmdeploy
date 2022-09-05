# Copyright (c) OpenMMLab. All rights reserved.

add_subdirectory(cpu)

if ("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
    add_subdirectory(cuda)
endif ()

if ("acl" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(acl)
endif ()
