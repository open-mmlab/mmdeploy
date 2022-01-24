# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy-example)

find_package(MMDeploy REQUIRED)
find_package(spdlog REQUIRED)

function(add_example name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE
          -Wl,--disable-new-dtags
          MMDeployLibs
          spdlog::spdlog)
  mmdeploy_load_static(${name} MMDeployStaticModules)
  mmdeploy_load_dynamic(${name} MMDeployDynamicModules)
endfunction()

add_example(image_classification)
add_example(object_detection)
add_example(image_restorer)
add_example(image_segmentation)
add_example(ocr)
