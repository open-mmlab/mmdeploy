# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.14)
project(mmdeploy_cxx_api)

if (MMDEPLOY_BUILD_SDK_CXX_API)
    add_library(${PROJECT_NAME} INTERFACE)
    target_include_directories(${PROJECT_NAME} INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>)
    foreach (task ${MMDEPLOY_TASKS})
        target_link_libraries(mmdeploy_${task} INTERFACE ${PROJECT_NAME})
    endforeach ()
    mmdeploy_export(${PROJECT_NAME})
endif ()
