# Copyright (c) OpenMMLab. All rights reserved.

if (MMDEPLOY_BUILD_SDK_PYTHON_API)
    set(MMDEPLOY_BUILD_SDK_CXX_API ON)
endif ()

add_subdirectory(c)
add_subdirectory(cxx)
add_subdirectory(java)
add_subdirectory(python)
