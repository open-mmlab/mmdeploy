# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_mmocr)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} OCR_SRCS)
aux_source_directory(${CMAKE_SOURCE_DIR}/third_party/clipper CLIPPER_SRCS)
set(SRCS ${OCR_SRCS} ${CLIPPER_SRCS})

mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")

add_subdirectory(cpu)
add_subdirectory(cuda)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/clipper)
target_link_libraries(${PROJECT_NAME} PRIVATE
        mmdeploy_opencv_utils
        mmdeploy_operation
        mmdeploy::transform
        mmdeploy::core)
add_library(mmdeploy::mmocr ALIAS ${PROJECT_NAME})

set(MMDEPLOY_TASKS ${MMDEPLOY_TASKS} text_detector text_recognizer CACHE INTERNAL "")
