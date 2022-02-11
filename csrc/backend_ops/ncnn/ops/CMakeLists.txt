# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_ncnn_ops)

include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)

# add plugin source
file(GLOB_RECURSE NCNN_OPS_SRCS *.cpp)
add_library(${PROJECT_NAME}_obj OBJECT "${NCNN_OPS_SRCS}")
set_target_properties(${PROJECT_NAME}_obj PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_include_directories(${PROJECT_NAME}_obj PUBLIC
        $<TARGET_PROPERTY:ncnn,INTERFACE_INCLUDE_DIRECTORIES>)
set(_COMMON_INCLUDE_DIRS
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/csrc>)
target_include_directories(${PROJECT_NAME}_obj
        PUBLIC ${_COMMON_INCLUDE_DIRS})
mmdeploy_export(${PROJECT_NAME}_obj)

mmdeploy_add_library(${PROJECT_NAME} SHARED EXCLUDE "")
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_obj)
target_include_directories(${PROJECT_NAME}
        PUBLIC ${_COMMON_INCLUDE_DIRS})

add_library(mmdeploy::ncnn_ops ALIAS ${PROJECT_NAME})
