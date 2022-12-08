# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_transform_module)

add_subdirectory(transform)

if (MMDEPLOY_ELENA_FUSION)
    add_subdirectory(elena)
endif ()

mmdeploy_add_module(${PROJECT_NAME} transform_module.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE mmdeploy::transform)
add_library(mmdeploy::transform_module ALIAS ${PROJECT_NAME})
