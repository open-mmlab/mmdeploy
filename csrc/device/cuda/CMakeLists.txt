# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)

include(${CMAKE_SOURCE_DIR}/cmake/cuda.cmake NO_POLICY_SCOPE)
project(mmdeploy_cuda_device CUDA CXX)

include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)

set(SRCS
        cuda_device.cpp
        cuda_builtin_kernels.cu)
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE cudart cuda)
add_library(mmdeploy::device::cuda ALIAS ${PROJECT_NAME})
