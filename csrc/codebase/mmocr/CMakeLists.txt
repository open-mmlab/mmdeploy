# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_mmocr)

include(${CMAKE_SOURCE_DIR}/cmake/opencv.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
set_targets(${PROJECT_NAME} MMOCR_OBJ MMOCR_STATIC MMOCR_SHARED)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} OCR_SRCS)
aux_source_directory(${CMAKE_SOURCE_DIR}/third_party/clipper CLIPPER_SRCS)
set(SRCS ${OCR_SRCS} ${CLIPPER_SRCS})
build_object_target(${MMOCR_OBJ} "${SRCS}")
target_include_directories(${MMOCR_OBJ} PRIVATE ${CMAKE_SOURCE_DIR}/third_party/clipper)
target_link_libraries(${MMOCR_OBJ} PRIVATE
        mmdeploy::core::static
        opencv_core opencv_imgproc
        )

build_static_target(${MMOCR_STATIC} ${MMOCR_OBJ} "PRIVATE")
add_library(mmdeploy::mmocr::static ALIAS ${MMOCR_STATIC})

build_shared_target(${MMOCR_SHARED} ${MMOCR_OBJ} "PRIVATE")
add_library(mmdeploy::mmocr ALIAS ${MMOCR_SHARED})

export_module(${MMOCR_STATIC} ${MMOCR_SHARED} ${MMOCR_OBJ})
