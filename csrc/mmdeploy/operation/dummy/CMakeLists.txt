# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_operation_dummy)

set(SRCS operations.cpp)

mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")

target_link_libraries(${PROJECT_NAME} PRIVATE mmdeploy_operation)
