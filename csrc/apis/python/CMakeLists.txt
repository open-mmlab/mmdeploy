# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_python)

if (NOT TARGET pybind11)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 pybind11)
endif ()

pybind11_add_module(mmdeploy_python ${CMAKE_CURRENT_SOURCE_DIR}/pydetector.cpp)
target_link_libraries(mmdeploy_python PRIVATE
        mmdeploy_detector
        -Wl,--whole-archive ${MMDEPLOY_STATIC_MODULES} -Wl,--no-whole-archive
        -Wl,--no-as-needed ${MMDEPLOY_DYNAMIC_MODULES} -Wl,--as-need
        ${MMDEPLOY_LIBS})
