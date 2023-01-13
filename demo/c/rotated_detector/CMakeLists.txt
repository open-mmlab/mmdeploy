# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_capi_rotated_detector)

if (NOT (${CMAKE_PROJECT_NAME} STREQUAL "MMDeploy"))
    find_package(MMDeploy REQUIRED)
endif ()

find_package(OpenCV REQUIRED)

add_executable(rotated_object_detection main.cpp)

if (NOT (MSVC OR APPLE))
    # Disable new dtags so that executables can run even without LD_LIBRARY_PATH set
    target_link_libraries(rotated_object_detection PRIVATE -Wl,--disable-new-dtags)
endif ()
if (MMDEPLOY_BUILD_SDK_MONOLITHIC)
    target_link_libraries(rotated_object_detection PRIVATE mmdeploy ${OpenCV_LIBS})
else ()
    # Load MMDeploy modules
    mmdeploy_load_static(rotated_object_detection MMDeployStaticModules)
    mmdeploy_load_dynamic(rotated_object_detection MMDeployDynamicModules)
    # Link to MMDeploy libraries
    target_link_libraries(rotated_object_detection PRIVATE MMDeployLibs ${OpenCV_LIBS})
endif ()
