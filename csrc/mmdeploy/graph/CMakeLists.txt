# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_graph)

set(SRCS
        profiler.cpp
        task.cpp
        static_router.cpp
        inference.cpp
        pipeline.cpp
        cond.cpp)
mmdeploy_add_module(${PROJECT_NAME} LIBRARY "${SRCS}")
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/concurrentqueue>)
add_library(mmdeploy::graph ALIAS ${PROJECT_NAME})
