# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_cxxapi_segmentor)

if (NOT (${CMAKE_PROJECT_NAME} STREQUAL "MMDeploy"))
    find_package(MMDeploy REQUIRED)
endif ()

find_package(OpenCV)

add_executable(segmentor main.cpp)

if (NOT (MSVC OR APPLE))
    # Disable new dtags so that executables can run even without LD_LIBRARY_PATH set
    target_link_libraries(segmentor PRIVATE -Wl,--disable-new-dtags)
endif ()
if (MMDEPLOY_BUILD_SDK_MONOLITHIC)
    target_link_libraries(segmentor PRIVATE mmdeploy ${OpenCV_LIBS})
else ()
    # Load MMDeploy modules
    mmdeploy_load_static(segmentor MMDeployStaticModules)
    mmdeploy_load_dynamic(segmentor MMDeployDynamicModules)
    # Link to MMDeploy libraries
    target_link_libraries(segmentor PRIVATE MMDeployLibs ${OpenCV_LIBS})
endif ()
