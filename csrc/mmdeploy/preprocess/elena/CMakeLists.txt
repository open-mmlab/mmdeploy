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
        dynamic_library.cpp
        library_compiler.cpp)
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
target_link_libraries(${PROJECT_NAME}
        PRIVATE mmdeploy::transform)
add_library(mmdeploy::transform_impl::elena ALIAS ${PROJECT_NAME})
