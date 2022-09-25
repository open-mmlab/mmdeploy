# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy-example)

if (NOT (${CMAKE_PROJECT_NAME} STREQUAL "MMDeploy"))
    find_package(MMDeploy REQUIRED)
endif ()

function(add_example_c task name)
    if (TARGET mmdeploy_${task})
        # Search for c/cpp sources
        file(GLOB _SRCS c/${name}.c*)
        add_executable(${name} ${_SRCS})
        if (NOT (MSVC OR APPLE))
            # Disable new dtags so that executables can run even without LD_LIBRARY_PATH set
            target_link_libraries(${name} PRIVATE -Wl,--disable-new-dtags)
        endif ()
        if (MMDEPLOY_BUILD_SDK_MONOLITHIC)
            target_link_libraries(${name} PRIVATE mmdeploy ${OpenCV_LIBS})
        else ()
            # Load MMDeploy modules
            mmdeploy_load_static(${name} MMDeployStaticModules)
            mmdeploy_load_dynamic(${name} MMDeployDynamicModules)
            # Link to MMDeploy libraries
            target_link_libraries(${name} PRIVATE MMDeployLibs ${OpenCV_LIBS})
        endif ()
        install(TARGETS ${name} RUNTIME DESTINATION bin/c)
    endif ()
endfunction()

function(add_example_cpp task name)
    if (TARGET mmdeploy_${task})
        # Search for c/cpp sources
        file(GLOB _SRCS cpp/${name}.c*)
        add_executable(${name} ${_SRCS})
        if (NOT (MSVC OR APPLE))
            # Disable new dtags so that executables can run even without LD_LIBRARY_PATH set
            target_link_libraries(${name} PRIVATE -Wl,--disable-new-dtags)
        endif ()
        if (MMDEPLOY_BUILD_SDK_MONOLITHIC)
            target_link_libraries(${name} PRIVATE mmdeploy ${OpenCV_LIBS})
        else ()
            # Load MMDeploy modules
            mmdeploy_load_static(${name} MMDeployStaticModules)
            mmdeploy_load_dynamic(${name} MMDeployDynamicModules)
            # Link to MMDeploy libraries
            target_link_libraries(${name} PRIVATE MMDeployLibs ${OpenCV_LIBS})
        endif ()
        install(TARGETS ${name} RUNTIME DESTINATION bin/cpp)
    endif ()
endfunction()

add_example_c(classifier image_classification)
add_example_c(detector object_detection)
add_example_c(segmentor image_segmentation)
add_example_c(restorer image_restorer)
add_example_c(text_detector ocr)
add_example_c(pose_detector pose_detection)
add_example_c(rotated_detector rotated_object_detection)

if (MMDEPLOY_BUILD_SDK_CXX_API)
    add_example_cpp(classifier classifier)
    add_example_cpp(detector detector)
    add_example_cpp(segmentor segmentor)
    add_example_cpp(restorer restorer)
    add_example_cpp(text_detector text_ocr)
    add_example_cpp(pose_detector pose_detector)
    add_example_cpp(rotated_detector rotated_detector)
endif ()
