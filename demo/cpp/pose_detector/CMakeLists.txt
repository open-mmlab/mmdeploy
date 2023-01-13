# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_cxxapi_pose_detector)

if (NOT (${CMAKE_PROJECT_NAME} STREQUAL "MMDeploy"))
    find_package(MMDeploy REQUIRED)
endif ()

find_package(OpenCV)

add_executable(pose_detector main.cpp)

if (NOT (MSVC OR APPLE))
    # Disable new dtags so that executables can run even without LD_LIBRARY_PATH set
    target_link_libraries(pose_detector PRIVATE -Wl,--disable-new-dtags)
endif ()
if (MMDEPLOY_BUILD_SDK_MONOLITHIC)
    target_link_libraries(pose_detector PRIVATE mmdeploy ${OpenCV_LIBS})
else ()
    # Load MMDeploy modules
    mmdeploy_load_static(pose_detector MMDeployStaticModules)
    mmdeploy_load_dynamic(pose_detector MMDeployDynamicModules)
    # Link to MMDeploy libraries
    target_link_libraries(pose_detector PRIVATE MMDeployLibs ${OpenCV_LIBS})
endif ()
