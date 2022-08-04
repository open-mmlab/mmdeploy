# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_snpe_net)

add_library(snpe SHARED IMPORTED)

if(NOT DEFINED ENV{MMDEPLOY_SNPE_X86_CI})
  set(sub_dir "aarch64-android-clang6.0")
else()
  set(sub_dir "x86_64-linux-clang")
endif()

if (NOT EXISTS $ENV{SNPE_ROOT}/lib/${sub_dir}/)
  message(ERROR "SNPE_ROOT directory not exist:  $ENV{SNPE_ROOT}/lib/${sub_dir}/")
endif()
message(STATUS "SNPE lib directory $ENV{SNPE_ROOT}/lib/${sub_dir}/")

set_target_properties(snpe PROPERTIES
  IMPORTED_LOCATION "$ENV{SNPE_ROOT}/lib/${sub_dir}/libSNPE.so"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{SNPE_ROOT}/include/zdl"
)

mmdeploy_add_module(${PROJECT_NAME} snpe_net.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE snpe)
add_library(mmdeploy::snpe_net ALIAS ${PROJECT_NAME})
