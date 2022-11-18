# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_operation)

set(SRCS operation.cpp vision.cpp)
mmdeploy_add_module(${PROJECT_NAME} LIBRARY "${SRCS}")

add_subdirectory(cpu)
add_subdirectory(cuda)
add_subdirectory(dummy)

add_library(mmdeploy::operation ALIAS ${PROJECT_NAME})
