# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_ncnn_net)

if("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES)
  include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)

  find_package(ncnn REQUIRED)
  add_library(${PROJECT_NAME} SHARED ncnn_net.cpp)
  target_link_libraries(${PROJECT_NAME} PRIVATE mmdeploy::core ncnn)
  target_link_libraries(
    ${PROJECT_NAME} PRIVATE -Wl,--whole-archive mmdeploy::ncnn_ops::static
                               -Wl,--no-whole-archive)
  add_library(mmdeploy::ncnn_net ALIAS ${PROJECT_NAME})
  export_module(${PROJECT_NAME})
else()
  message(
    ERROR
    "'ncnn_net' is NOT supported in target devices: ${MMDEPLOY_TARGET_DEVICES}")
endif()
