# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_ncnn_net)

if ("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES)

    find_package(ncnn REQUIRED)

    mmdeploy_add_module(${PROJECT_NAME} ncnn_net.cpp)
    target_link_libraries(${PROJECT_NAME} PRIVATE mmdeploy_ncnn_ops_obj)
    target_link_libraries(${PROJECT_NAME} PRIVATE ncnn)
    add_library(mmdeploy::ncnn_net ALIAS ${PROJECT_NAME})
else ()
    message(
            ERROR
            "'ncnn_net' is NOT supported in target devices: ${MMDEPLOY_TARGET_DEVICES}")
endif ()
