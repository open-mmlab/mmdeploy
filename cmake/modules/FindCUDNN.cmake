# Copyright (c) OpenMMLab. All rights reserved.

if (NOT DEFINED CUDNN_DIR)
    set(CUDNN_DIR $ENV{CUDNN_DIR})
endif ()

find_path(
    CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDNN_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(
    CUDNN_LIBRARY_CUDNN_PATH cudnn
    HINTS ${CUDNN_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

if (NOT (CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY_CUDNN_PATH))
    message(FATAL_ERROR "Couldn't find cuDNN in CUDNN_DIR: ${CUDNN_DIR}, "
        "or in CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}, "
        "please check if the path is correct.")
endif()

add_library(cudnn SHARED IMPORTED)
set_property(TARGET cudnn APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
if (MSVC)
    set_target_properties(cudnn PROPERTIES
        IMPORTED_IMPLIB_RELEASE ${CUDNN_LIBRARY_CUDNN_PATH}
        INTERFACE_INCLUDE_DIRECTORIES ${CUDNN_INCLUDE_DIR}
    )

else()
    set_target_properties(cudnn PROPERTIES
        IMPORTED_LOCATION_RELEASE ${CUDNN_LIBRARY_CUDNN_PATH}
        INTERFACE_INCLUDE_DIRECTORIES ${CUDNN_INCLUDE_DIR}
    )
endif()
