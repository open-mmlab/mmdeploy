# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_ort_net)

include(${CMAKE_SOURCE_DIR}/cmake/modules/FindONNXRUNTIME.cmake)

mmdeploy_add_net(${PROJECT_NAME} ort_net.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE mmdeploy_onnxruntime_ops_obj)
target_link_libraries(${PROJECT_NAME} PUBLIC onnxruntime)
add_library(mmdeploy::ort_net ALIAS ${PROJECT_NAME})
