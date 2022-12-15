# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy-example)

if (NOT (${CMAKE_PROJECT_NAME} STREQUAL "MMDeploy"))
    find_package(MMDeploy REQUIRED)
endif ()


function(add_example dep folder name)
    if (NOT dep OR TARGET mmdeploy_${dep})
        # Search for c/cpp sources
        file(GLOB _SRCS ${folder}/${name}.c*)
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
        install(TARGETS ${name} RUNTIME DESTINATION bin)
    endif ()
endfunction()

add_example(classifier c image_classification)
add_example(classifier c batch_image_classification)
add_example(detector c object_detection)
add_example(detector c batch_object_detection)
add_example(segmentor c image_segmentation)
add_example(restorer c image_restorer)
add_example(text_detector c ocr)
add_example(pose_detector c pose_detection)
add_example(rotated_detector c rotated_object_detection)
add_example(video_recognizer c video_recognition)
# TODO: figure out a better way
# add_example("" c det_cls)
# add_example("" c det_pose)

if (MMDEPLOY_BUILD_SDK_CXX_API)
    add_example(classifier cpp classifier)
    add_example(detector cpp detector)
    add_example(segmentor cpp segmentor)
    add_example(restorer cpp restorer)
    add_example(text_detector cpp text_ocr)
    add_example(text_detector cpp text_det_recog)
    add_example(pose_detector cpp pose_detector)
    add_example(rotated_detector cpp rotated_detector)
    add_example(pose_detector cpp pose_tracker)
    add_example(video_recognizer cpp video_cls)
endif ()
