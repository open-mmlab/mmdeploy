# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_capi_classifier)

if (NOT (${CMAKE_PROJECT_NAME} STREQUAL "MMDeploy"))
    find_package(MMDeploy REQUIRED)
endif ()

find_package(OpenCV)

macro(link_mmdeploy target_name)
    if (NOT (MSVC OR APPLE))
        # Disable new dtags so that executables can run even without LD_LIBRARY_PATH set
        target_link_libraries(${target_name} PRIVATE -Wl,--disable-new-dtags)
    endif ()
    if (MMDEPLOY_BUILD_SDK_MONOLITHIC)
        target_link_libraries(${target_name} PRIVATE mmdeploy ${OpenCV_LIBS})
    else ()
        # Load MMDeploy modules
        mmdeploy_load_static(${target_name} MMDeployStaticModules)
        mmdeploy_load_dynamic(${target_name} MMDeployDynamicModules)
        # Link to MMDeploy libraries
        target_link_libraries(${target_name} PRIVATE MMDeployLibs ${OpenCV_LIBS})
    endif ()
endmacro()

add_executable(image_classification main.cpp)
add_executable(batch_image_classification batch.cpp)

link_mmdeploy(image_classification)
link_mmdeploy(batch_image_classification)
