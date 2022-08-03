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

file(GLOB CPU_KERNEL_SRCS "cpu_kernel/*.cpp")

set(ALL_SRCS ${SRCS} ${CPU_KERNEL_SRCS})
if ("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
        file(GLOB CUDA_KERNEL_SRCS "cuda_kernel/*.cu")
        set(ALL_SRCS ${ALL_SRCS} ${CUDA_KERNEL_SRCS})
endif ()

mmdeploy_add_module(${PROJECT_NAME} "${ALL_SRCS}")
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME}
        PRIVATE mmdeploy::transform)
if ("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
        target_link_libraries(${PROJECT_NAME} PRIVATE cuda)
endif ()
add_library(mmdeploy::transform_impl::elena ALIAS ${PROJECT_NAME})
