# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy-example)

find_package(MMDeploy REQUIRED)

function(add_example name)
  file(GLOB _SRCS ${name}.c*)
  add_executable(${name} ${_SRCS})
  if (NOT MSVC)
    # disable new dtags so that executables can run even without LD_LIBRARY_PATH set
    target_link_libraries(${name} PRIVATE -Wl,--disable-new-dtags)
  endif ()
  mmdeploy_load_static(${name} MMDeployStaticModules)
  mmdeploy_load_dynamic(${name} MMDeployDynamicModules)
  target_link_libraries(${name} PRIVATE MMDeployLibs ${OpenCV_LIBS})
endfunction()

add_example(image_classification)
add_example(object_detection)
add_example(image_restorer)
add_example(image_segmentation)
add_example(pose_detection)
add_example(ocr)
