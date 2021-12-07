# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_pplnn_net)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
find_package(pplnn REQUIRED)

set_targets(${PROJECT_NAME} PPL_NET_OBJ PPL_NET_STATIC PPL_NET_SHARED)


build_object_target(${PPL_NET_OBJ} ppl_net.cpp)
target_include_directories(${PPL_NET_OBJ} PUBLIC
        $<BUILD_INTERFACE:${PPLNN_INCLUDE_DIRS}>)

target_link_libraries(${PPL_NET_OBJ} PRIVATE
        mmdeploy::core::static
        ${PPLNN_LIBRARIES}
        )

if ("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES)
    target_compile_definitions(${PPL_NET_OBJ} PRIVATE -DPPL_NN_HAS_X86=1)
endif ()

if ("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
    target_compile_definitions(${PPL_NET_OBJ} PRIVATE -DPPL_NN_HAS_CUDA=1)
    target_include_directories(${PPL_NET_OBJ} PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/include)
    target_link_directories(${PPL_NET_OBJ} PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    target_link_libraries(${PPL_NET_OBJ}
            PUBLIC nvrtc)
endif ()

build_static_target(${PPL_NET_STATIC} ${PPL_NET_OBJ} "PUBLIC")

add_library(mmdeploy::pplnn_net::static ALIAS ${PPL_NET_STATIC})

build_shared_target(${PPL_NET_SHARED} ${PPL_NET_OBJ} "PRIVATE")
add_library(mmdeploy::pplnn_net ALIAS ${PPL_NET_SHARED})

export_module(${PPL_NET_STATIC} ${PPL_NET_SHARED} ${PPL_NET_OBJ})
