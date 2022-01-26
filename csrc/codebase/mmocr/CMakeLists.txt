# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_mmocr)

include(${CMAKE_SOURCE_DIR}/cmake/opencv.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} OCR_SRCS)
aux_source_directory(${CMAKE_SOURCE_DIR}/third_party/clipper CLIPPER_SRCS)
set(SRCS ${OCR_SRCS} ${CLIPPER_SRCS})
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/third_party/clipper)
target_link_libraries(${PROJECT_NAME}
        PRIVATE mmdeploy::transform mmdeploy_opencv_utils)
add_library(mmdeploy::mmocr ALIAS ${PROJECT_NAME})
