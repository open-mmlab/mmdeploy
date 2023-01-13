# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_capi_demo)

if (NOT (${CMAKE_PROJECT_NAME} STREQUAL "MMDeploy"))
    find_package(MMDeploy REQUIRED)
endif ()


function(add_example dep name)
    if (NOT dep OR TARGET mmdeploy_${dep})
        add_executable(${name} ${name}.cpp)
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
#        install(TARGETS ${name} RUNTIME DESTINATION bin/c)
    endif ()
endfunction()

add_example(classifier image_classification)
add_example(classifier batch_image_classification)
add_example(detector object_detection)
add_example(detector batch_object_detection)
add_example(segmentor image_segmentation)
add_example(restorer image_restorer)
add_example(text_detector ocr)
add_example(pose_detector pose_detection)
add_example(rotated_detector rotated_object_detection)
add_example(video_recognizer video_recognition)
