# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.14)
project(mmdeploy_runtime)

set(MMDEPLOY_RUNTIME_SRCS
        common.cpp
        internal.cpp
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
    list(APPEND MMDEPLOY_RUNTIME_SRCS ${task_name}.cpp)
endforeach ()

pybind11_add_module(${PROJECT_NAME} ${MMDEPLOY_RUNTIME_SRCS})
# disable MMDEPLOY_CXX_USE_OPENCV in apis/cxx/mmdeploy/common.hpp
target_compile_definitions(${PROJECT_NAME} PRIVATE -DMMDEPLOY_CXX_USE_OPENCV=0)
if (APPLE)
        set_target_properties(${PROJECT_NAME} PROPERTIES
                BUILD_RPATH "@loader_path"
                INSTALL_RPATH "@loader_path")
else ()
        set_target_properties(${PROJECT_NAME} PROPERTIES
                BUILD_RPATH "\$ORIGIN"
                INSTALL_RPATH "\$ORIGIN")
endif ()

# https://github.com/pybind/pybind11/issues/1604
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -fsized-deallocation)
endif ()

if (MMDEPLOY_BUILD_SDK_MONOLITHIC)
    target_link_libraries(${PROJECT_NAME} PRIVATE mmdeploy)
else ()
    mmdeploy_load_static(${PROJECT_NAME} MMDeployStaticModules)
    mmdeploy_load_dynamic(${PROJECT_NAME} MMDeployDynamicModules)
    target_link_libraries(${PROJECT_NAME} PRIVATE MMDeployLibs)
endif ()

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/demo/python/ DESTINATION example/python)
