# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy-example)

find_package(OpenCV REQUIRED)
find_package(MMDeploy REQUIRED)

set(MMDeploy_LIBS ${MMDEPLOY_LIBS}
    -Wl,--no-as-needed ${MMDEPLOY_DYNAMIC_MODULES} -Wl,--as-needed
    -Wl,--whole-archive ${MMDEPLOY_STATIC_MODULES} -Wl,--no-whole-archive
    -Wl,--disable-new-dtags)

function(add_example name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} ${MMDeploy_LIBS} opencv_imgcodecs
          opencv_imgproc opencv_core)
endfunction()

add_example(image_classification)
add_example(object_detection)
add_example(image_restorer)
add_example(image_segmentation)
add_example(ocr)
