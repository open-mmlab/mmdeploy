# Copyright (c) OpenMMLab. All rights reserved.
if (NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "MMDeploy's installation directory")
endif ()
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

cmake_minimum_required(VERSION 3.14)
project(MMDeploy VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# options
option(BUILD_SHARED_LIBS "build shared lib" ON)
option(MMDEPLOY_BUILD_MODEL_OPTIMIZER "build MMDeploy's model optimizer" ON)
option(MMDEPLOY_BUILD_SDK "build MMDeploy's SDK" OFF)
option(MMDEPLOY_ZIP_MODEL "support sdk model in zip format" OFF)
option(MMDEPLOY_BUILD_TEST "build MMDeploy's csrc's unittest" OFF)
set(MMDEPLOY_TARGET_DEVICES
        "cpu" CACHE STRING "MMDeploy's target devices")
set(MMDEPLOY_TARGET_BACKENDS "" CACHE STRING "MMDeploy's target inference engines")
set(MMDEPLOY_CODEBASES "" CACHE STRING "select OpenMMLab's codebases")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "choose 'Release' as default build type" FORCE)
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

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
# set INTERFACE target to gather linked modules
add_library(MMDeployStaticModules INTERFACE)
add_library(MMDeployDynamicModules INTERFACE)
add_library(MMDeployLibs INTERFACE)

add_subdirectory(csrc)

if (MMDEPLOY_BUILD_SDK)
    # get static modules and dynamic modules from ${MMDeployStaticModules} and ${MMDeployDynamicModules}, respectively
    set(STATIC_MODULES)
    get_target_property(STATIC_MODULES MMDeployStaticModules INTERFACE_LINK_LIBRARIES)
    get_target_list("${STATIC_MODULES}" FILTERED_MODULES)
    set(MMDEPLOY_STATIC_MODULES "${FILTERED_MODULES}" CACHE STRING "MMDeploy's static modules")
    message(STATUS "MMDEPLOY_STATIC_MODULES: ${MMDEPLOY_STATIC_MODULES}")

    set(DYNAMIC_MODULES)
    get_target_property(DYNAMIC_MODULES MMDeployDynamicModules INTERFACE_LINK_LIBRARIES)
    get_target_list("${DYNAMIC_MODULES}" FILTERED_MODULES)
    set(MMDEPLOY_DYNAMIC_MODULES "${FILTERED_MODULES}" CACHE STRING "MMDeploy's dynamic modules")
    message(STATUS "MMDEPLOY_DYNAMIC_MODULES: ${MMDEPLOY_DYNAMIC_MODULES}")

    # get libs from ${MMDeployLibs}
    set(LIBS)
    get_target_property(LIBS MMDeployLibs INTERFACE_LINK_LIBRARIES)
    get_target_list("${LIBS}" FILTERED_LIBS)
    set(MMDEPLOY_LIBS "${FILTERED_LIBS}" CACHE STRING "MMDeploy's libs that can be linked directly by application")
    message(STATUS "MMDEPLOY_LIBS: ${MMDEPLOY_LIBS}")

    if (MMDEPLOY_BUILD_TEST)
        add_subdirectory(tests/test_csrc)
    endif ()

    if (MMDEPLOY_BUILD_SDK_PYTHON_API)
        add_subdirectory(csrc/apis/python)
    endif()

    # export MMDeploy package
    install(EXPORT MMDeployTargets
            # NAMESPACE mmdeploy::
            FILE MMDeployTargets.cmake
            #EXPORT_LINK_INTERFACE_LIBRARIES
            DESTINATION lib/cmake/MMDeploy)

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
            DESTINATION lib/cmake/MMDeploy
            )

    install(DIRECTORY ${CMAKE_SOURCE_DIR}/demo/csrc/ DESTINATION example)
endif ()
