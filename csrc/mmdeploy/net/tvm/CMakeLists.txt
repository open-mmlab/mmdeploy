# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_tvm_net)

include(${CMAKE_SOURCE_DIR}/cmake/modules/FindTVM.cmake)

mmdeploy_add_module(${PROJECT_NAME} tvm_net.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${TVM_INCLUDE_DIR} ${DLPACK_INCLUDE_DIR} ${DMLC_CORE_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE tvm_runtime mmdeploy_dlpack_utils)

add_library(mmdeploy::tvm_net ALIAS ${PROJECT_NAME})
