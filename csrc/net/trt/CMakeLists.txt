# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_trt_net)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tensorrt.cmake)

add_library(${PROJECT_NAME} SHARED trt_net.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
        ${TENSORRT_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDNN_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_directories(${PROJECT_NAME} PUBLIC ${CUDNN_DIR}/lib64)
target_link_libraries(${PROJECT_NAME} PUBLIC ${TENSORRT_LIBRARY} cudnn)

target_link_libraries(${PROJECT_NAME}
        PRIVATE mmdeploy::core
        )
target_link_libraries(${PROJECT_NAME}
        PRIVATE -Wl,--whole-archive
            mmdeploy::tensorrt_ops::static
        -Wl,--no-whole-archive
        )
add_library(mmdeploy::trt_net ALIAS ${PROJECT_NAME})
export_module(${PROJECT_NAME})
