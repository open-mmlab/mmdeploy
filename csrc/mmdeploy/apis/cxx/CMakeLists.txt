# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.14)
project(mmdeploy_cxx_api)

if (MMDEPLOY_BUILD_SDK_CXX_API)
    foreach (task_name ${MMDEPLOY_TASKS})
        set(_target_name mmdeploy_cxx_${task_name})
        add_library(${_target_name} OBJECT mmdeploy/${task_name}.cpp)
        set_target_properties(${_target_name} PROPERTIES POSITION_INDEPENDENT_CODE 1)
        target_include_directories(${_target_name} PRIVATE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../c>)
        target_link_libraries(${_target_name} PUBLIC mmdeploy::core)
        target_link_libraries(mmdeploy_${task_name} PRIVATE ${_target_name})
        target_include_directories(mmdeploy_${task_name} PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
        mmdeploy_export(${_target_name})
    endforeach ()
endif ()
