# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_graph)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
set_targets(${PROJECT_NAME} GRAPH_OBJ GRAPH_STATIC GRAPH_SHARED)

set(SRCS
        common.cpp
        inference.cpp
        pipeline.cpp
        task.cpp
        flatten.cpp
        unflatten.cpp)
build_object_target(${GRAPH_OBJ} "${SRCS}")
target_link_libraries(${GRAPH_OBJ} PRIVATE mmdeploy::core::static)

build_static_target(${GRAPH_STATIC} ${GRAPH_OBJ} "PRIVATE")
add_library(mmdeploy::graph::static ALIAS ${GRAPH_STATIC})

build_shared_target(${GRAPH_SHARED} ${GRAPH_OBJ} "PRIVATE")
add_library(mmdeploy::graph ALIAS ${GRAPH_SHARED})

export_module(${GRAPH_STATIC} ${GRAPH_SHARED} ${GRAPH_OBJ})
