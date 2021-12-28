# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_python)

if (NOT TARGET pybind11)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 pybind11)
endif ()

pybind11_add_module(mmdeploy_python
        ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/classifier.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/detector.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/segmentor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/text_detector.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/text_recognizer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/restorer.cpp)

target_link_libraries(mmdeploy_python PRIVATE
        mmdeploy_classifier
        mmdeploy_detector
        mmdeploy_segmentor
        mmdeploy_text_detector
        mmdeploy_text_recognizer
        mmdeploy_restorer
        ${MMDEPLOY_LIBS}
        -Wl,--whole-archive ${MMDEPLOY_STATIC_MODULES} -Wl,--no-whole-archive
        -Wl,--no-as-needed ${MMDEPLOY_DYNAMIC_MODULES} -Wl,--as-need
        )

target_include_directories(mmdeploy_python PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../..)