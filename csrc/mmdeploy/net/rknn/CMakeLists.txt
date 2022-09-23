# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_rknn_net)

add_library(rknn SHARED IMPORTED)

if (NOT EXISTS $ENV{RKNN_ROOT}/runtime/RK3588/Linux/librknn_api/aarch64/)
  message(ERROR "RKNN_ROOT directory not exist:  $ENV{RKNN_ROOT}/runtime/RK3588/Linux/librknn_api/aarch64/")
endif()
message(STATUS "RKNN lib directory $ENV{RKNN_ROOT}/runtime/RK3588/Linux/librknn_api/aarch64/")

set_target_properties(rknn PROPERTIES
  IMPORTED_LOCATION "$ENV{RKNN_ROOT}/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{RKNN_ROOT}/runtime/RK3588/Linux/librknn_api/include"
)

mmdeploy_add_module(${PROJECT_NAME} rknn_net.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE rknn)
add_library(mmdeploy::rknn_net ALIAS ${PROJECT_NAME})
