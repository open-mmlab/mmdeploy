# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_cuda_transform_impl CUDA CXX)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
  # suppress 'CMAKE_CUDA_ARCHITECTURES' warning
  cmake_policy(SET CMP0104 OLD)
endif()

find_package(pplcv REQUIRED)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
# include(${CMAKE_SOURCE_DIR}/cmake/cuda.cmake NO_POLICY_SCOPE)
set_targets(${PROJECT_NAME} TRANSFORM_OBJ TRANSFORM_STATIC TRANSFORM_SHARED)

set(SRCS
    crop_impl.cpp
    image2tensor_impl.cpp
    load_impl.cpp
    normalize_impl.cpp
    pad_impl.cpp
    resize_impl.cpp
    cast.cu
    crop.cu
    normalize.cu
    transpose.cu)
build_object_target(${TRANSFORM_OBJ} "${SRCS}")

target_include_directories(${TRANSFORM_OBJ}
                           PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries(
  ${TRANSFORM_OBJ} PRIVATE ${PPLCV_LIBRARIES} mmdeploy::transform::static
                           mmdeploy::core::static)

build_static_target(${TRANSFORM_STATIC} ${TRANSFORM_OBJ} "PUBLIC")
add_library(mmdeploy::transform_impl::cuda::static ALIAS ${TRANSFORM_STATIC})

build_shared_target(${TRANSFORM_SHARED} ${TRANSFORM_OBJ} "PUBLIC")
add_library(mmdeploy::transform_impl::cuda ALIAS ${TRANSFORM_SHARED})

export_module(${TRANSFORM_STATIC} ${TRANSFORM_SHARED} ${TRANSFORM_OBJ})
