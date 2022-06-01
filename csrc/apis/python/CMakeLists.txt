# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_python)

if ("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
    include(${CMAKE_SOURCE_DIR}/cmake/cuda.cmake)
endif ()

if (NOT TARGET pybind11)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 pybind11)
endif ()

set(MMDEPLOY_PYTHON_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/executor.cpp)

macro(mmdeploy_python_add_module name)
    if (TARGET mmdeploy_${name})
        list(APPEND MMDEPLOY_PYTHON_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp)
    endif ()
endmacro()

mmdeploy_python_add_module(classifier)
mmdeploy_python_add_module(detector)
mmdeploy_python_add_module(segmentor)
mmdeploy_python_add_module(text_detector)
mmdeploy_python_add_module(text_recognizer)
mmdeploy_python_add_module(restorer)
mmdeploy_python_add_module(pose_detector)
mmdeploy_python_add_module(rotated_detector)

pybind11_add_module(${PROJECT_NAME} ${MMDEPLOY_PYTHON_SRCS})

mmdeploy_load_static(${PROJECT_NAME} MMDeployStaticModules)
mmdeploy_load_dynamic(${PROJECT_NAME} MMDeployDynamicModules)
target_link_libraries(${PROJECT_NAME} PRIVATE MMDeployLibs)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../..)
