# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_net_module)

set(BACKEND_LIB_NAMES)

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

if ("rknn" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(rknn)
endif ()

if ("tvm" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(tvm)
endif ()

if (MMDEPLOY_DYNAMIC_BACKEND)
    set(_MODULE_STR ${BACKEND_LIB_NAMES})
    list(TRANSFORM _MODULE_STR REPLACE "(.+)" "\"\\1\"")
    string(JOIN ",\n        " _MODULE_STR ${_MODULE_STR})
    set(_MMDEPLOY_DYNAMIC_MODULES ${_MODULE_STR})

    set(_LOADER_NAME net_loader)
    set(_LOADER_PATH ${${PROJECT_NAME}_BINARY_DIR}/${_LOADER_NAME}.cpp)
    configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/loader.cpp.in
        ${_LOADER_PATH})
    if (NOT (WIN32 OR APPLE))
        SET(_DL_LIB dl)
    endif ()
endif ()

mmdeploy_add_module(${PROJECT_NAME} net_module.cpp ${_LOADER_PATH})
target_link_libraries(${PROJECT_NAME} PUBLIC ${_DL_LIB})
add_library(mmdeploy::net_module ALIAS ${PROJECT_NAME})
