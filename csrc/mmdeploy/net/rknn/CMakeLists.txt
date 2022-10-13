# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_rknn_net)

add_library(rknn SHARED IMPORTED)

if(DEFINED ENV{RKNPU2_DEVICE_DIR})
    file(TO_CMAKE_PATH $ENV{RKNPU2_DEVICE_DIR} RKNPU2_DEVICE_DIR)
else()
    message(FATAL_ERROR "RKNPU2_DEVICE_DIR env must be defined")
endif()

set_target_properties(rknn PROPERTIES
  IMPORTED_LOCATION "${RKNPU2_DEVICE_DIR}/Linux/librknn_api/aarch64/librknn_api.so"
  INTERFACE_INCLUDE_DIRECTORIES "${RKNPU2_DEVICE_DIR}/Linux/librknn_api/include"
)

mmdeploy_add_module(${PROJECT_NAME} rknn_net.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE rknn)
add_library(mmdeploy::rknn_net ALIAS ${PROJECT_NAME})
