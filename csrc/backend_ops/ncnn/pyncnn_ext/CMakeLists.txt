# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(ncnn_ext)

# pybind11
if (NOT TARGET pybind11)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 pybind11)
endif ()

pybind11_add_module(ncnn_ext ncnn_ext.cpp)

target_link_libraries(ncnn_ext PUBLIC mmdeploy_ncnn_ops ncnn)
set_target_properties(
        ncnn_ext PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_SOURCE_DIR}/mmdeploy/backend/ncnn)
