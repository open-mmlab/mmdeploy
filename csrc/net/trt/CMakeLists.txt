# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_trt_net)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tensorrt.cmake)

set_targets("mmdeploy_trt_net" TRT_NET_OBJ TRT_NET_STATIC TRT_NET_SHARED)

build_object_target(${TRT_NET_OBJ} trt_net.cpp)
target_include_directories(${TRT_NET_OBJ} PRIVATE
        ${TENSORRT_INCLUDE_DIR})
target_include_directories(${TRT_NET_OBJ} PRIVATE ${CUDNN_DIR}/include)
target_include_directories(${TRT_NET_OBJ} PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_directories(${TRT_NET_OBJ} PUBLIC ${CUDNN_DIR}/lib64)
target_link_libraries(${TRT_NET_OBJ} PUBLIC ${TENSORRT_LIBRARY} cudnn)

target_link_libraries(${TRT_NET_OBJ}
        PRIVATE mmdeploy::core::static
        )

build_static_target(${TRT_NET_STATIC} ${TRT_NET_OBJ} "PUBLIC")
add_library(mmdeploy::trt_net::static ALIAS ${TRT_NET_STATIC})

build_shared_target(${TRT_NET_SHARED} ${TRT_NET_OBJ} "PUBLIC")
target_link_libraries(${TRT_NET_SHARED}
        PRIVATE -Wl,--whole-archive
            mmdeploy::tensorrt_ops::static
        -Wl,--no-whole-archive
        )
add_library(mmdeploy::trt_net ALIAS ${TRT_NET_SHARED})

export_module(${TRT_NET_STATIC} ${TRT_NET_SHARED} ${TRT_NET_OBJ})
export_module(mmdeploy_tensorrt_ops_static "" mmdeploy_tensorrt_ops_obj)
