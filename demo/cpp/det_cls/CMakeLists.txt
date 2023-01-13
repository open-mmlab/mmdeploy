# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_cxxapi_det_cls)

if (NOT (${CMAKE_PROJECT_NAME} STREQUAL "MMDeploy"))
    find_package(MMDeploy REQUIRED)
endif ()

find_package(OpenCV)

add_executable(det_cls main.cpp)

if (NOT (MSVC OR APPLE))
    # Disable new dtags so that executables can run even without LD_LIBRARY_PATH set
    target_link_libraries(det_cls PRIVATE -Wl,--disable-new-dtags)
endif ()
if (MMDEPLOY_BUILD_SDK_MONOLITHIC)
    target_link_libraries(det_cls PRIVATE mmdeploy ${OpenCV_LIBS})
else ()
    # Load MMDeploy modules
    mmdeploy_load_static(det_cls MMDeployStaticModules)
    mmdeploy_load_dynamic(det_cls MMDeployDynamicModules)
    # Link to MMDeploy libraries
    target_link_libraries(det_cls PRIVATE MMDeployLibs ${OpenCV_LIBS})
endif ()
