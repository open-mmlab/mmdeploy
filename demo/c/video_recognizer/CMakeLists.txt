# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_cxxapi_video_recognizer)

if (NOT (${CMAKE_PROJECT_NAME} STREQUAL "MMDeploy"))
    find_package(MMDeploy REQUIRED)
endif ()

find_package(OpenCV)

add_executable(video_recognition main.cpp)

if (NOT (MSVC OR APPLE))
    # Disable new dtags so that executables can run even without LD_LIBRARY_PATH set
    target_link_libraries(video_recognition PRIVATE -Wl,--disable-new-dtags)
endif ()
if (MMDEPLOY_BUILD_SDK_MONOLITHIC)
    target_link_libraries(video_recognition PRIVATE mmdeploy ${OpenCV_LIBS})
else ()
    # Load MMDeploy modules
    mmdeploy_load_static(video_recognition MMDeployStaticModules)
    mmdeploy_load_dynamic(video_recognition MMDeployDynamicModules)
    # Link to MMDeploy libraries
    target_link_libraries(video_recognition PRIVATE MMDeployLibs ${OpenCV_LIBS})
endif ()
