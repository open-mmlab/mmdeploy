# Copyright (c) OpenMMLab. All rights reserved.

project(ncnn_ext)

# pybind11
if (NOT TARGET pybind11)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 pybind11)
endif ()

pybind11_add_module(ncnn_ext ncnn_ext.cpp)

target_link_libraries(ncnn_ext PUBLIC mmdeploy_ncnn_ops ncnn)
set(_NCNN_EXT_DIR ${CMAKE_SOURCE_DIR}/mmdeploy/backend/ncnn)
set_target_properties(ncnn_ext PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${_NCNN_EXT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${_NCNN_EXT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${_NCNN_EXT_DIR})
