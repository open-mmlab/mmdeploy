# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_core)

set(SPDLOG_LIB)

if (MMDEPLOY_SPDLOG_EXTERNAL)
    find_package(spdlog QUIET)
    if (spdlog_FOUND)
        set(SPDLOG_LIB spdlog::spdlog)
    endif ()
else ()
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/spdlog
            ${CMAKE_CURRENT_BINARY_DIR}/spdlog EXCLUDE_FROM_ALL)
    set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
    mmdeploy_export(spdlog)
    set(SPDLOG_LIB spdlog::spdlog)
    if (MMDEPLOY_BUILD_SDK_CPP_API)
        install(DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/spdlog DESTINATION include/cpp)
    endif ()
endif ()

include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)

set(SRCS
        device_impl.cpp
        logger.cpp
        graph.cpp
        mat.cpp
        model.cpp
        module.cpp
        net.cpp
        operator.cpp
        status_code.cpp
        tensor.cpp
        registry.cpp
        utils/device_utils.cpp
        utils/formatter.cpp
        utils/stacktrace.cpp)

mmdeploy_add_library(${PROJECT_NAME} ${SRCS})
target_compile_definitions(${PROJECT_NAME} PUBLIC -DMMDEPLOY_STATUS_USE_SOURCE_LOCATION=1)

target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/csrc>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/outcome>
        # TODO: remove dependency of `json`
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/json>
        )

if (MMDEPLOY_BUILD_SDK_CPP_API)
    target_include_directories(${PROJECT_NAME} PUBLIC
            $<INSTALL_INTERFACE:include/cpp>)
endif ()

target_link_libraries(${PROJECT_NAME} PUBLIC ${SPDLOG_LIB})
if (NOT (MSVC OR ANDROID))
    target_link_libraries(${PROJECT_NAME} PUBLIC stdc++fs)
endif ()
add_library(mmdeploy::core ALIAS ${PROJECT_NAME})

if (MMDEPLOY_BUILD_SDK_CPP_API)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/csrc/core
            DESTINATION include/cpp
            FILES_MATCHING PATTERN "*.h")
    install(FILES ${CMAKE_SOURCE_DIR}/third_party/outcome/outcome-experimental.hpp
            DESTINATION include/cpp/core)

    install(DIRECTORY ${CMAKE_SOURCE_DIR}/csrc/experimental
            DESTINATION include/cpp
            FILES_MATCHING PATTERN "*.h")
endif ()
