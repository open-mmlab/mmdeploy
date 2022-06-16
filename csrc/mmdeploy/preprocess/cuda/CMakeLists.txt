# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_cuda_transform_impl CUDA CXX)

find_package(pplcv REQUIRED)

set(SRCS
        crop_impl.cpp
        image2tensor_impl.cpp
        default_format_bundle_impl.cpp
        load_impl.cpp
        normalize_impl.cpp
        pad_impl.cpp
        resize_impl.cpp
        cast.cu
        crop.cu
        normalize.cu
        transpose.cu)
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
target_link_libraries(${PROJECT_NAME} PRIVATE
        mmdeploy::transform ${PPLCV_LIBRARIES})
target_include_directories(${PROJECT_NAME}
        PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include ${PPLCV_INCLUDE_DIRS})
add_library(mmdeploy::transform_impl::cuda ALIAS ${PROJECT_NAME})
