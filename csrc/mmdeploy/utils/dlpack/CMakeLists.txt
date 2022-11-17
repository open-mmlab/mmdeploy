# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_dlpack_utils)

mmdeploy_add_library(${PROJECT_NAME} STATIC dlpack_utils.cpp)

target_link_libraries(${PROJECT_NAME}
        PRIVATE mmdeploy::core)

target_include_directories(${PROJECT_NAME}
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_include_directories(${PROJECT_NAME} PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/dlpack>)
