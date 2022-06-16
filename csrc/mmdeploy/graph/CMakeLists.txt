# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_graph)

set(SRCS
        common.cpp
        task.cpp
        pipeline.cpp
        inference.cpp
        flatten.cpp
        unflatten.cpp)
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
add_library(mmdeploy::graph ALIAS ${PROJECT_NAME})
