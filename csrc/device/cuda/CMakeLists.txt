# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_cuda_device)

set(SRCS
        cuda_device.cpp
        cuda_builtin_kernels.cu)
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES} cuda)
add_library(mmdeploy::device::cuda ALIAS ${PROJECT_NAME})
