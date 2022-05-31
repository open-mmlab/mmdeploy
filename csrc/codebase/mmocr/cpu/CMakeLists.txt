# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.14)
project(mmdeploy_mmocr_cpu_impl CXX)

if ("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES)
    include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)
    add_library(${PROJECT_NAME} OBJECT dbnet.cpp panet.cpp psenet.cpp)
    set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE 1)
    target_link_libraries(${PROJECT_NAME} PRIVATE
            mmdeploy_opencv_utils
            mmdeploy::core)
    target_link_libraries(mmdeploy_mmocr PRIVATE ${PROJECT_NAME})
    mmdeploy_export(${PROJECT_NAME})
endif ()
