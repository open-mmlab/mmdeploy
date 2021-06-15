cmake_minimum_required (VERSION 3.10)
project (mmdeploy_backend_ops)

# TensorRT config

# enable tensorrt
option(BUILD_TENSORRT_OPS "enable tensorrt ops" OFF)
# TensorRT search path
if (BUILD_TENSORRT_OPS)
    if (NOT DEFINED TENSORRT_DIR)
    set(TENSORRT_DIR  $ENV{TENSORRT_DIR})
    endif()
endif()

add_subdirectory (backend_ops)
