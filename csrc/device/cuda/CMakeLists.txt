# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)

include(${CMAKE_SOURCE_DIR}/cmake/cuda.cmake NO_POLICY_SCOPE)
project(mmdeploy_cuda_device CUDA CXX)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
    # suppress 'CMAKE_CUDA_ARCHITECTURES' warning
    cmake_policy(SET CMP0104 OLD)
endif ()

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
set_targets(${PROJECT_NAME} CUDA_DEVICE_OBJ CUDA_DEVICE_STATIC CUDA_DEVICE_SHARED)

set(SRCS
        cuda_device.cpp
        cuda_builtin_kernels.cu)
build_target(${PROJECT_NAME} "${SRCS}")
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_directories(${PROJECT_NAME} PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
target_link_libraries(${PROJECT_NAME}
        PRIVATE mmdeploy::core
        PUBLIC cudart cuda)
add_library(mmdeploy::device::cuda ALIAS ${PROJECT_NAME})
export_module(${PROJECT_NAME})
