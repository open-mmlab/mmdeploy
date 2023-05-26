if (NOT MSVC)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif ()

# build ONNXRUNTIME ops
if ("ort" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    if (NOT DEFINED ONNXRUNTIME_DIR)
        set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
    endif ()
    if (NOT ONNXRUNTIME_DIR)
        message(FATAL_ERROR " ONNXRUNTIME_DIR is not found.")
    else ()
        message(STATUS "Build ONNXRUNTIME custom ops.")
        add_subdirectory(onnxruntime)
    endif ()
endif ()

# build TensorRT ops
if ("trt" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    if (NOT DEFINED TENSORRT_DIR)
        set(TENSORRT_DIR $ENV{TENSORRT_DIR})
    endif ()
    message(STATUS "Build TensorRT custom ops.")
    add_subdirectory(tensorrt)
endif ()

# build ncnn ops
if ("ncnn" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    message(STATUS "Build ncnn custom ops")
    add_subdirectory(ncnn)
endif ()

# build TorchScript ops
if ("torchscript" IN_LIST MMDEPLOY_TARGET_BACKENDS
    OR "coreml" IN_LIST MMDEPLOY_TARGET_BACKENDS)
  message(STATUS "Build torchscript custom ops")
  add_subdirectory(torchscript)
endif ()
