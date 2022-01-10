# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_python)

if (NOT TARGET pybind11)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 pybind11)
endif ()

set(MMDEPLOY_PYTHON_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp)

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

pybind11_add_module(${PROJECT_NAME} ${MMDEPLOY_PYTHON_SRCS})

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${MMDEPLOY_LIBS}
    -Wl,--whole-archive ${MMDEPLOY_STATIC_MODULES} -Wl,--no-whole-archive
    -Wl,--no-as-needed ${MMDEPLOY_DYNAMIC_MODULES} -Wl,--as-need)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)
