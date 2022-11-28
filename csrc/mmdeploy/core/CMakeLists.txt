# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_core)

# this is used to keep compatibility with legacy spdlog where CMake package is not available
set(SPDLOG_LIB)

if (MMDEPLOY_SPDLOG_EXTERNAL)
    find_package(spdlog QUIET)
    if (spdlog_FOUND)
        set(SPDLOG_LIB spdlog::spdlog)
    endif ()
else ()
    set(MMDEPLOY_SPDLOG_DIR ${CMAKE_SOURCE_DIR}/third_party/spdlog)
    add_subdirectory(${MMDEPLOY_SPDLOG_DIR} ${CMAKE_CURRENT_BINARY_DIR}/spdlog EXCLUDE_FROM_ALL)
    set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if (NOT (MMDEPLOY_SHARED_LIBS OR MSVC))
        target_compile_options(spdlog PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)
    endif ()
    set(SPDLOG_LIB spdlog::spdlog)
    mmdeploy_export(spdlog)
    if (MMDEPLOY_BUILD_SDK_CXX_API)
        install(DIRECTORY ${MMDEPLOY_SPDLOG_DIR}/include/spdlog
                DESTINATION include/mmdeploy/third_party)
    endif ()
endif ()

set(SRCS
        device_impl.cpp
        logger.cpp
        mat.cpp
        model.cpp
        module.cpp
        net.cpp
        operator.cpp
        status_code.cpp
        tensor.cpp
        registry.cpp
        graph.cpp
        utils/device_utils.cpp
        utils/formatter.cpp
        utils/stacktrace.cpp
        profiler.cpp
        )

mmdeploy_add_library(${PROJECT_NAME} ${SRCS})

target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/csrc>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/outcome>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/concurrentqueue>
        # TODO: remove dependency of `json`
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/json>
        )

if (MMDEPLOY_STATUS_USE_STACKTRACE)
    include(${CMAKE_SOURCE_DIR}/cmake/stacktrace.cmake)
else ()
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DMMDEPLOY_STATUS_USE_SOURCE_LOCATION=1)
endif ()

if (MMDEPLOY_BUILD_SDK_CXX_API)
    target_include_directories(${PROJECT_NAME} PUBLIC
            $<INSTALL_INTERFACE:include>
            $<INSTALL_INTERFACE:include/mmdeploy/third_party/outcome>
            $<INSTALL_INTERFACE:include/mmdeploy/third_party/json>)
    if (NOT MMDEPLOY_SPDLOG_EXTERNAL)
        target_include_directories(spdlog INTERFACE
                $<INSTALL_INTERFACE:include/mmdeploy/third_party>)
    endif ()
endif ()

target_link_libraries(${PROJECT_NAME} PUBLIC ${SPDLOG_LIB})

include(${CMAKE_SOURCE_DIR}/cmake/filesystem.cmake)
if (STD_FS_LIB)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${STD_FS_LIB})
endif ()

add_library(mmdeploy::core ALIAS ${PROJECT_NAME})

if (MMDEPLOY_BUILD_SDK_CXX_API)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/csrc/mmdeploy/core
            DESTINATION include/mmdeploy
            FILES_MATCHING PATTERN "*.h")
    install(FILES ${CMAKE_SOURCE_DIR}/third_party/outcome/outcome-experimental.hpp
            DESTINATION include/mmdeploy/third_party/outcome)

    install(DIRECTORY ${CMAKE_SOURCE_DIR}/csrc/mmdeploy/experimental
            DESTINATION include/mmdeploy
            FILES_MATCHING PATTERN "*.h")
endif ()
