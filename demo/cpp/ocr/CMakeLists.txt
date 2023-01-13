# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_cxxapi_ocr)

if (NOT (${CMAKE_PROJECT_NAME} STREQUAL "MMDeploy"))
    find_package(MMDeploy REQUIRED)
endif ()

find_package(OpenCV REQUIRED)

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

add_executable(ocr main.cpp)
add_executable(ocr2 text_det_recog.cpp)

link_mmdeploy(ocr)
link_mmdeploy(ocr2)
