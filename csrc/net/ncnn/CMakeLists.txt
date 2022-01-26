# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_ncnn_net)

if ("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES)
    include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)

    find_package(ncnn REQUIRED)

    mmdeploy_add_module(${PROJECT_NAME} ncnn_net.cpp)
    target_link_libraries(${PROJECT_NAME} PRIVATE ncnn)
    if (BUILD_SHARED_LIBS)
        mmdeploy_load_dynamic(${PROJECT_NAME} mmdeploy::ncnn_ops)
    else ()
        mmdeploy_load_static(${PROJECT_NAME} mmdeploy::ncnn_ops::static)
    endif ()
    add_library(mmdeploy::ncnn_net ALIAS ${PROJECT_NAME})
else ()
    message(
            ERROR
            "'ncnn_net' is NOT supported in target devices: ${MMDEPLOY_TARGET_DEVICES}")
endif ()
