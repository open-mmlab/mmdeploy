# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)

add_subdirectory(cpu)

if ("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
    add_subdirectory(cuda)
endif ()
