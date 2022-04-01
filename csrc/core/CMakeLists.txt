# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_core)

#[==[
Refer to: https://cmake.org/cmake/help/latest/command/file.html
Note We do not recommend using GLOB to collect a list of source files from your source tree.
If no CMakeLists.txt file changes when a source is added or removed then the generated build
system cannot know when to ask CMake to regenerate. The CONFIGURE_DEPENDS flag may not work
reliably on all generators, or if a new generator is added in the future that cannot support it,
projects using it will be stuck. Even if CONFIGURE_DEPENDS works reliably, there is still a cost
to perform the check on every rebuild.
#]==]
set(SPDLOG_LIB)
find_package(spdlog QUIET)
if (spdlog_FOUND)
    message(STATUS "spdlog is found")
    set(SPDLOG_LIB spdlog::spdlog)
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
        utils/stacktrace.cpp
        ../experimental/execution/execution.cpp
        ../experimental/execution/timed_single_thread_context.cpp)

mmdeploy_add_library(${PROJECT_NAME} ${SRCS})
target_compile_definitions(${PROJECT_NAME} PUBLIC -DMMDEPLOY_STATUS_USE_SOURCE_LOCATION=1)

target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/csrc>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/outcome>
        # TODO: remove dependency of `json`
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/json>
        $<INSTALL_INTERFACE:include/cpp>
        )
target_link_libraries(${PROJECT_NAME} PUBLIC ${SPDLOG_LIB})
if (NOT (MSVC OR ANDROID))
    target_link_libraries(${PROJECT_NAME} PUBLIC stdc++fs)
endif ()
add_library(mmdeploy::core ALIAS ${PROJECT_NAME})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/csrc/core
        DESTINATION include/cpp
        FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_SOURCE_DIR}/third_party/outcome/outcome-experimental.hpp
        DESTINATION include/cpp/core)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/csrc/experimental
        DESTINATION include/cpp
        FILES_MATCHING PATTERN "*.h")
