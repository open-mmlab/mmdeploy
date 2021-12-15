# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_ncnn_net)

if("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES)
  include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)

  set_targets(${PROJECT_NAME} NCNN_NET_OBJ NCNN_NET_STATIC NCNN_NET_SHARED)

  find_package(ncnn REQUIRED)

  build_object_target(${NCNN_NET_OBJ} ncnn_net.cpp)
  target_link_libraries(${NCNN_NET_OBJ} PRIVATE mmdeploy::core::static ncnn)

  build_static_target(${NCNN_NET_STATIC} ${NCNN_NET_OBJ} "PUBLIC")
  add_library(mmdeploy::ncnn_net::static ALIAS ${NCNN_NET_STATIC})

  build_shared_target(${NCNN_NET_SHARED} ${NCNN_NET_OBJ} "PRIVATE")
  target_link_libraries(
    ${NCNN_NET_SHARED} PRIVATE -Wl,--whole-archive mmdeploy::ncnn_ops::static
                               -Wl,--no-whole-archive)
  add_library(mmdeploy::ncnn_net ALIAS ${NCNN_NET_SHARED})

  export_module(${NCNN_NET_STATIC} ${NCNN_NET_SHARED} ${NCNN_NET_OBJ})
else()
  message(
    ERROR
    "'ncnn_net' is NOT supported in target devices: ${MMDEPLOY_TARGET_DEVICES}")
endif()
