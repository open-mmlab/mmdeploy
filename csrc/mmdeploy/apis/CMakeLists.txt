# Copyright (c) OpenMMLab. All rights reserved.

# Python API depends on C++ API
if (MMDEPLOY_BUILD_SDK_PYTHON_API)
    set(MMDEPLOY_BUILD_SDK_CXX_API ON)
endif ()

add_subdirectory(c)
add_subdirectory(cxx)
add_subdirectory(java)

# add python subdir conditionally since it's designed to work as
# a standalone project also
if (MMDEPLOY_BUILD_SDK_PYTHON_API)
    add_subdirectory(python)
endif ()
