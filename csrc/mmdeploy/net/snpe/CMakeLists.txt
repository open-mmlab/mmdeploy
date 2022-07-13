# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_snpe_net)

add_library(snpe SHARED IMPORTED)

if (NOT EXISTS $ENV{SNPE_ROOT}/lib/aarch64-android-clang6.0/)
  message(ERROR "SNPE_ROOT directory not exist:  $ENV{SNPE_ROOT}/lib/aarch64-android-clang6.0/")
endif()

set_target_properties(snpe PROPERTIES
  IMPORTED_LOCATION "$ENV{SNPE_ROOT}/lib/aarch64-android-clang6.0/libSNPE.so"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{SNPE_ROOT}/include/zdl"
)

message(STATUS ">> project name ${PROJECT_NAME}")
mmdeploy_add_module(${PROJECT_NAME} snpe_net.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE snpe)
add_library(mmdeploy::snpe_net ALIAS ${PROJECT_NAME})
