# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_capi_segmentor)

if (NOT (${CMAKE_PROJECT_NAME} STREQUAL "MMDeploy"))
    find_package(MMDeploy REQUIRED)
endif ()

find_package(OpenCV REQUIRED)

add_executable(image_segmentation main.cpp)

if (NOT (MSVC OR APPLE))
    # Disable new dtags so that executables can run even without LD_LIBRARY_PATH set
    target_link_libraries(image_segmentation PRIVATE -Wl,--disable-new-dtags)
endif ()
if (MMDEPLOY_BUILD_SDK_MONOLITHIC)
    target_link_libraries(image_segmentation PRIVATE mmdeploy ${OpenCV_LIBS})
else ()
    # Load MMDeploy modules
    mmdeploy_load_static(image_segmentation MMDeployStaticModules)
    mmdeploy_load_dynamic(image_segmentation MMDeployDynamicModules)
    # Link to MMDeploy libraries
    target_link_libraries(image_segmentation PRIVATE MMDeployLibs ${OpenCV_LIBS})
endif ()
