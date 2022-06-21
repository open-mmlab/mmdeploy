# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_elena_transform_impl)

set(SRCS
        crop_impl.cpp
        collect_impl.cpp
        image2tensor_impl.cpp
        default_format_bundle_impl.cpp
        load_impl.cpp
        normalize_impl.cpp
        pad_impl.cpp
        resize_impl.cpp
        elena_registry.cpp)

file(GLOB KERNEL_SRCS "prebuilt_kernel/*.c" "prebuilt_kernel/*.cpp")
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}" "${KERNEL_SRCS}")
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME}
        PRIVATE mmdeploy::transform)
add_library(mmdeploy::transform_impl::elena ALIAS ${PROJECT_NAME})
