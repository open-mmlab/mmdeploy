# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_archive)

add_library(${PROJECT_NAME} INTERFACE)
target_link_libraries(${PROJECT_NAME} INTERFACE mmdeploy::core)
add_library(mmdeploy::archive ALIAS mmdeploy_archive)

if (MMDEPLOY_BUILD_SDK_CXX_API)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/csrc/archive
            DESTINATION include/cpp
            FILES_MATCHING PATTERN "*.h")
    install(FILES ${CMAKE_SOURCE_DIR}/third_party/json/json.hpp
            DESTINATION include/cpp/archive)
endif ()
