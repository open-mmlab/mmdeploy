# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(javaapis)

include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)
include_directories(${CMAKE_SOURCE_DIR}/include/c)
include_directories(${CMAKE_SOURCE_DIR}/include/cpp/core)
include_directories(${CMAKE_SOURCE_DIR}/include/cpp/experimental)
find_package(OpenCV REQUIRED core imgproc)
find_package(ncnn REQUIRED)
find_package(spdlog REQUIRED)
message(STATUS "spdlog lib dir: ${spdlog_DIR}/../../")
link_directories(${spdlog_DIR}/../../)

macro(mmdeploy_java_add_module name)
    list(APPEND MMDEPLOY_JAVA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp)
endmacro()

mmdeploy_java_add_module(classifier)
mmdeploy_java_add_module(detector)
mmdeploy_java_add_module(segmentor)
mmdeploy_java_add_module(restorer)
mmdeploy_java_add_module(text_detector)
mmdeploy_java_add_module(text_recognizer)
mmdeploy_java_add_module(pose_detector)
add_library(${PROJECT_NAME} SHARED ${MMDEPLOY_JAVA_SRCS})

mmdeploy_load_static(${PROJECT_NAME} MMDeployStaticModules)
mmdeploy_load_dynamic(${PROJECT_NAME} MMDeployDynamicModules)
target_link_libraries(${PROJECT_NAME} PRIVATE MMDeployLibs PUBLIC ncnn
        ${OpenCV_LIBS} spdlog)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../..)
