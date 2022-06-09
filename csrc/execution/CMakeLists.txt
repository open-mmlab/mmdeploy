# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_execution)

set(SRCS schedulers/schedulers.cpp)
mmdeploy_add_module(${PROJECT_NAME} LIBRARY "${SRCS}")
add_library(mmdeploy::execution ALIAS ${PROJECT_NAME})
