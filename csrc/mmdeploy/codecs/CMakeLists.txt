# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_decoder)

add_subdirectory(nvjpeg)
add_subdirectory(nvdec)

mmdeploy_add_module(${PROJECT_NAME} LIBRARY decoder.cpp)
add_library(mmdeploy::decoder ALIAS ${PROJECT_NAME})
