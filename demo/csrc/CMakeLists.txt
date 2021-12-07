# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy-example)

find_package(OpenCV REQUIRED)
find_package(MMDeploy REQUIRED)

function(add_example name libs)

    add_executable(${name} ${name}.cpp)

    target_link_libraries(${name}
            ${libs}
        -Wl,--no-as-needed
            ${MMDeploy_DYNAMIC_MODULES}
        -Wl,--as-needed
        opencv_imgcodecs opencv_imgproc opencv_core)
endfunction()

add_example(image_classification "mmdeploy_classifier")
add_example(object_detection "mmdeploy_detector")
