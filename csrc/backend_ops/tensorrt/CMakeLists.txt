# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
include(${CMAKE_SOURCE_DIR}/cmake/cuda.cmake NO_POLICY_SCOPE)
project(mmdeploy_tensorrt_ops CUDA CXX)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/cuda.cmake NO_POLICY_SCOPE)
include(${CMAKE_SOURCE_DIR}/cmake/tensorrt.cmake NO_POLICY_SCOPE)

set_targets(${PROJECT_NAME} BACKEND_OPS_OBJ BACKEND_OPS_STATIC BACKEND_OPS_MODULE)

# cub
if (NOT DEFINED CUB_ROOT_DIR)
    if (CUDA_VERSION VERSION_LESS 11.0)
        set(CUB_ROOT_DIR "${PROJECT_SOURCE_DIR}/third_party/cub")
    endif()
endif ()

file(GLOB_RECURSE BACKEND_OPS_SRCS *.cpp *.cu)
build_object_target(${BACKEND_OPS_OBJ} "${BACKEND_OPS_SRCS}")
target_compile_definitions(${BACKEND_OPS_OBJ}
        PRIVATE -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT=1)
target_include_directories(${BACKEND_OPS_OBJ}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_include_directories(${BACKEND_OPS_OBJ}
        PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_include_directories(${BACKEND_OPS_OBJ} PRIVATE ${TENSORRT_INCLUDE_DIR})
target_include_directories(${BACKEND_OPS_OBJ} PRIVATE ${CUDNN_DIR}/include)
target_include_directories(${BACKEND_OPS_OBJ} PRIVATE ${CUB_ROOT_DIR})
target_link_directories(${BACKEND_OPS_OBJ} PUBLIC ${CUDNN_DIR}/lib64)
target_link_libraries(${BACKEND_OPS_OBJ}
        PRIVATE ${TENSORRT_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} cudnn)

# Build module library. It is used to convert onnx model to tensorrt engine
build_module_target(${BACKEND_OPS_MODULE} ${BACKEND_OPS_OBJ} "PRIVATE")
add_library(mmdeploy::tensorrt_ops ALIAS ${BACKEND_OPS_MODULE})
install_targets(${BACKEND_OPS_MODULE})

if (MMDEPLOY_BUILD_SDK)
    ## Build static library. SDK's uses it to build `trt_net` module
    build_static_target(${BACKEND_OPS_STATIC} ${BACKEND_OPS_OBJ} "PRIVATE")
    add_library(mmdeploy::tensorrt_ops::static ALIAS ${BACKEND_OPS_STATIC})
endif ()
