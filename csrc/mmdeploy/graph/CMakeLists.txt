# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_graph)

set(SRCS
        common.cpp
        task.cpp
        static_router.cpp
        inference.cpp
        flatten.cpp
        unflatten.cpp
        flattened.cpp
        pipeline.cpp)
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
add_library(mmdeploy::graph ALIAS ${PROJECT_NAME})
