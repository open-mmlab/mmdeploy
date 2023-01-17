# Copyright (c) OpenMMLab. All rights reserved.
if (NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "installation directory")
endif ()
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

cmake_minimum_required(VERSION 3.14)
project(MMDeploy VERSION 0.12.0)

set(CMAKE_CXX_STANDARD 17)

set(MMDEPLOY_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(MMDEPLOY_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(MMDEPLOY_VERSION_PATCH ${PROJECT_VERSION_PATCH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
if (MSVC)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
else ()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif ()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# options
option(MMDEPLOY_SHARED_LIBS "build shared libs" OFF)
option(MMDEPLOY_BUILD_SDK "build MMDeploy SDK" OFF)
option(MMDEPLOY_BUILD_SDK_MONOLITHIC "build single lib for SDK API" ON)
option(MMDEPLOY_BUILD_TEST "build unittests" OFF)
option(MMDEPLOY_BUILD_SDK_PYTHON_API "build SDK Python API" OFF)
option(MMDEPLOY_BUILD_SDK_CXX_API "build SDK C++ API" OFF)
option(MMDEPLOY_BUILD_SDK_CSHARP_API "build SDK C# API support" OFF)
option(MMDEPLOY_BUILD_SDK_JAVA_API "build SDK JAVA API" OFF)
option(MMDEPLOY_BUILD_EXAMPLES "build examples" OFF)
option(MMDEPLOY_SPDLOG_EXTERNAL "use external spdlog" OFF)
option(MMDEPLOY_ZIP_MODEL "support SDK model in zip format" OFF)
option(MMDEPLOY_COVERAGE "build SDK for coverage" OFF)
option(MMDEPLOY_ELENA_FUSION "use elena to fuse preprocess" OFF)

set(MMDEPLOY_TARGET_DEVICES "cpu" CACHE STRING "target devices to support")
set(MMDEPLOY_TARGET_BACKENDS "" CACHE STRING "target inference engines to support")
set(MMDEPLOY_CODEBASES "all" CACHE STRING "select OpenMMLab codebases")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "choose 'Release' as default build type" FORCE)
endif ()

if (MMDEPLOY_SHARED_LIBS)
    set(MMDEPLOY_LIB_TYPE SHARED)
else ()
    set(MMDEPLOY_LIB_TYPE STATIC)
endif ()

set(MMDEPLOY_TASKS "" CACHE INTERNAL "")

if (MMDEPLOY_COVERAGE)
    add_compile_options(-coverage -fprofile-arcs -ftest-coverage)
    add_link_options(-coverage -lgcov)
endif ()

# when CUDA devices are enabled, the environment variable ASAN_OPTIONS=protect_shadow_gap=0
# must be set at runtime
if (MMDEPLOY_ASAN_ENABLE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
    add_link_options(-fsanitize=address)
endif ()

# notice that ubsan has linker issues for ubuntu < 18.04, see
# https://stackoverflow.com/questions/50024731/ld-unrecognized-option-push-state-no-as-needed
if (MMDEPLOY_UBSAN_ENABLE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=undefined>)
    add_link_options(-fsanitize=undefined)
endif ()

if (MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/diagnostics:classic>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/wd4251>)
endif ()

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-arc")
endif()

add_library(MMDeployStaticModules INTERFACE)
add_library(MMDeployDynamicModules INTERFACE)
add_library(MMDeployLibs INTERFACE)

if ((cuda IN_LIST MMDEPLOY_TARGET_DEVICES) OR (trt IN_LIST MMDEPLOY_TARGET_BACKENDS))
    include(cmake/cuda.cmake NO_POLICY_SCOPE)
endif ()

# this must come after including cuda.cmake because policies in function scope is captured
# at function definition
include(cmake/MMDeploy.cmake)

add_subdirectory(csrc/mmdeploy)

if (MMDEPLOY_BUILD_SDK)
    install(TARGETS MMDeployStaticModules
            MMDeployDynamicModules
            MMDeployLibs
            EXPORT MMDeployTargets)

    if (MMDEPLOY_BUILD_TEST)
        add_subdirectory(tests/test_csrc)
    endif ()

    if (MMDEPLOY_BUILD_EXAMPLES)
        include(${CMAKE_SOURCE_DIR}/cmake/opencv.cmake)
        add_subdirectory(demo/csrc)
    endif ()

    # export MMDeploy package
    install(EXPORT MMDeployTargets
            FILE MMDeployTargets.cmake
            DESTINATION lib/cmake/MMDeploy)

    if (MMDEPLOY_SPDLOG_EXTERNAL)
        set(SPDLOG_DEPENDENCY "find_package(spdlog QUIET)")
    endif ()
    # append backend deps
    mmdeploy_add_deps(trt BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS TENSORRT CUDNN)
    mmdeploy_add_deps(ort BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS ONNXRUNTIME)
    mmdeploy_add_deps(ncnn BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS ncnn)
    mmdeploy_add_deps(openvino BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS InferenceEngine)
    if (NOT MMDEPLOY_SHARED_LIBS)
        mmdeploy_add_deps(pplnn BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS pplnn)
    endif ()
    mmdeploy_add_deps(snpe BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS snpe)
    mmdeploy_add_deps(rknn BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS rknn)

    include(CMakePackageConfigHelpers)
    # generate the config file that is includes the exports
    configure_package_config_file(${CMAKE_SOURCE_DIR}/cmake/MMDeployConfig.cmake.in
            "${CMAKE_CURRENT_BINARY_DIR}/MMDeployConfig.cmake"
            INSTALL_DESTINATION "lib/cmake"
            NO_SET_AND_CHECK_MACRO
            NO_CHECK_REQUIRED_COMPONENTS_MACRO
            )

    write_basic_package_version_file(
            "${CMAKE_CURRENT_BINARY_DIR}/MMDeployConfigVersion.cmake"
            VERSION "${MMDeploy_VERSION_MAJOR}.${MMDeploy_VERSION_MINOR}"
            COMPATIBILITY AnyNewerVersion
    )

    install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/MMDeployConfig.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/MMDeployConfigVersion.cmake
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MMDeploy.cmake
            DESTINATION lib/cmake/MMDeploy
            )

    if (MSVC)
        install(FILES
                ${CMAKE_CURRENT_SOURCE_DIR}/cmake/loader.cpp.in
                DESTINATION lib/cmake/MMDeploy
                )
    endif ()

    install(DIRECTORY
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules
            DESTINATION lib/cmake/MMDeploy
            )

    if (${CMAKE_VERSION} VERSION_LESS "3.17.0")
        install(SCRIPT cmake/post-install.cmake)
    endif ()
endif ()
