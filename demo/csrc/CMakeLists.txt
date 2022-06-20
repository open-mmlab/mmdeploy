# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy-example)

find_package(MMDeploy REQUIRED)

function(add_example task name)
  if (TARGET mmdeploy_${task})
    # Search for c/cpp sources
    file(GLOB _SRCS ${name}.c*)
    add_executable(${name} ${_SRCS})
    if (NOT MSVC)
      # Disable new dtags so that executables can run even without LD_LIBRARY_PATH set
      target_link_libraries(${name} PRIVATE -Wl,--disable-new-dtags)
    endif ()
    # Load MMDeploy modules
    mmdeploy_load_static(${name} MMDeployStaticModules)
    mmdeploy_load_dynamic(${name} MMDeployDynamicModules)
    # Link to MMDeploy libraries
    target_link_libraries(${name} PRIVATE MMDeployLibs ${OpenCV_LIBS})
  endif ()
endfunction()

add_example(classifier       image_classification)
add_example(detector         object_detection)
add_example(segmentor        image_segmentation)
add_example(restorer         image_restorer)
add_example(text_detector    ocr)
add_example(pose_detector    pose_detection)
add_example(rotated_detector rotated_object_detection)
