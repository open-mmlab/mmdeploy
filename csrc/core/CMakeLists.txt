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
# file(GLOB_RECURSE CORE_SRCS *.cpp)
set(SPDLOG_LIB)
find_package(spdlog QUIET)
if (spdlog_FOUND)
    message(STATUS "spdlog is found")
    set(SPDLOG_LIB spdlog::spdlog)
endif ()

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
set_targets(${PROJECT_NAME} CORE_OBJ CORE_STATIC CORE_SHARED)

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
        utils/formatter.cpp
        utils/stacktrace.cpp)
build_object_target(${CORE_OBJ} "${SRCS}")
target_compile_definitions(${CORE_OBJ} PUBLIC -DMM_STATUS_USE_SOURCE_LOCATION=1)
target_include_directories(${CORE_OBJ}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/csrc>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/outcome>
        # TODO(lzhang): remove dependency of `json`
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/json>
        $<INSTALL_INTERFACE:include/cpp>
        )
target_link_libraries(${CORE_OBJ} ${SPDLOG_LIB} stdc++fs)

build_static_target(${CORE_STATIC} ${CORE_OBJ} "PUBLIC")
add_library(mmdeploy::core::static ALIAS ${CORE_STATIC})

build_shared_target(${CORE_SHARED} ${CORE_OBJ} "PUBLIC")
add_library(mmdeploy::core ALIAS ${CORE_SHARED})

#install_targets("${CORE_STATIC};${CORE_SHARED}")

install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/core
        DESTINATION include/cpp
        FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_SOURCE_DIR}/third_party/outcome/outcome-experimental.hpp
        DESTINATION include/cpp/core)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/experimental
        DESTINATION include/cpp
        FILES_MATCHING PATTERN "*.h")

install(TARGETS ${CORE_STATIC} ${CORE_SHARED} ${CORE_OBJ}
        EXPORT MMDeployTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        )
