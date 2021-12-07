# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_ncnn_ops)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
set_targets(${PROJECT_NAME} NCNN_OPS_OBJ NCNN_OPS_STATIC NCNN_OPS_SHARED)

# add plugin source
file(GLOB_RECURSE NCNN_OPS_SRCS *.cpp)
build_object_target(${NCNN_OPS_OBJ} "${NCNN_OPS_SRCS}")
target_link_libraries(${NCNN_OPS_OBJ} ncnn)

build_shared_target(${NCNN_OPS_SHARED} ${NCNN_OPS_OBJ} "PRIVATE")
install_targets(${NCNN_OPS_SHARED})

if (MMDEPLOY_BUILD_SDK)
    ## Build static library. SDK's uses it to build `ncnn_net` module
    build_static_target(${NCNN_OPS_STATIC} ${NCNN_OPS_OBJ} "PRIVATE")
    add_library(mmdeploy::ncnn_ops::static ALIAS ${NCNN_OPS_STATIC})
endif ()
