# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_net_module)

if ("trt" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(trt)
endif ()

if ("pplnn" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(ppl)
endif ()

if ("ort" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(ort)
endif ()

if ("ncnn" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(ncnn)
endif ()

if ("openvino" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(openvino)
endif ()

if ("snpe" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(snpe)
endif ()

if ("acl" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(acl)
endif ()

if ("torchscript" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(torchscript)
endif ()

if ("coreml" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(coreml)
endif ()

mmdeploy_add_module(${PROJECT_NAME} net_module.cpp)
add_library(mmdeploy::net_module ALIAS ${PROJECT_NAME})
