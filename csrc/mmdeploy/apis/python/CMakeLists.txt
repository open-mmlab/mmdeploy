# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.14)
project(mmdeploy_python)

set(MMDEPLOY_PYTHON_SRCS
        common.cpp
        internal.cpp
        executor.cpp
        pipeline.cpp)

set(CMAKE_CXX_STANDARD 17)

if (${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    # standard alone project
    add_subdirectory(${CMAKE_SOURCE_DIR}/../../../../third_party/pybind11
            ${CMAKE_CURRENT_BINARY_DIR}/pybind11)
    find_package(MMDeploy REQUIRED)
elseif (NOT TARGET pybind11)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 pybind11)
endif ()

foreach (task_name ${MMDEPLOY_TASKS})
    list(APPEND MMDEPLOY_PYTHON_SRCS ${task_name}.cpp)
endforeach ()

pybind11_add_module(${PROJECT_NAME} ${MMDEPLOY_PYTHON_SRCS})


mmdeploy_load_static(${PROJECT_NAME} MMDeployStaticModules)
mmdeploy_load_dynamic(${PROJECT_NAME} MMDeployDynamicModules)
target_link_libraries(${PROJECT_NAME} PRIVATE MMDeployLibs)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/demo/python/ DESTINATION example/python)
