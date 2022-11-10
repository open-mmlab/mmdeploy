# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_graph)

set(SRCS
        task.cpp
        static_router.cpp
        inference.cpp
        pipeline.cpp
        cond.cpp)
mmdeploy_add_module(${PROJECT_NAME} LIBRARY "${SRCS}")
add_library(mmdeploy::graph ALIAS ${PROJECT_NAME})
