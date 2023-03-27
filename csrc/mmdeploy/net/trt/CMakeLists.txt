# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_trt_net)

include(${CMAKE_SOURCE_DIR}/cmake/tensorrt.cmake)

mmdeploy_add_net(${PROJECT_NAME} trt_net.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
        ${TENSORRT_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDNN_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE mmdeploy_tensorrt_ops_obj)
target_link_libraries(${PROJECT_NAME} PUBLIC ${TENSORRT_LIBS} cudnn cuda)

add_library(mmdeploy::trt_net ALIAS ${PROJECT_NAME})
