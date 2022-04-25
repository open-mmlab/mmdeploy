# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_archive)

add_library(${PROJECT_NAME} INTERFACE)
target_link_libraries(${PROJECT_NAME} INTERFACE mmdeploy::core)
add_library(mmdeploy::archive ALIAS mmdeploy_archive)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/csrc/archive
        DESTINATION include/cpp
        FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_SOURCE_DIR}/third_party/json/json.hpp DESTINATION include/cpp/archive)
