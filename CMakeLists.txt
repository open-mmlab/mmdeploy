cmake_minimum_required (VERSION 3.10)
project (mmdeploy_backend_ops)

# ONNXRUNTIME config

# enable onnxruntime
option(BUILD_ONNXRUNTIME_OPS "enable ONNXRUNTIME ops" OFF)
# ONNXRUNTIME search path
if (BUILD_ONNXRUNTIME_OPS)
    if (NOT DEFINED ONNXRUNTIME_DIR)
        set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
    endif()
    if (NOT ONNXRUNTIME_DIR)
    message(ERROR " ONNXRUNTIME_DIR is not found.")
    endif()
endif()

# TensorRT config

# enable tensorrt
option(BUILD_TENSORRT_OPS "enable TensorRT ops" OFF)
# TensorRT search path
if (BUILD_TENSORRT_OPS)
    if (NOT DEFINED TENSORRT_DIR)
        set(TENSORRT_DIR  $ENV{TENSORRT_DIR})
    endif()
endif()

add_subdirectory (backend_ops)
