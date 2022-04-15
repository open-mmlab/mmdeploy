# Copyright (c) OpenMMLab. All rights reserved.

if (NOT DEFINED TENSORRT_DIR)
    set(TENSORRT_DIR $ENV{TENSORRT_DIR})
endif ()
if (NOT TENSORRT_DIR)
    message(FATAL_ERROR "Please set TENSORRT_DIR with cmake -D option.")
endif()

find_path(
    TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

if (NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "Cannot find TensorRT header NvInfer.h "
        "in TENSORRT_DIR: ${TENSORRT_DIR} or in CUDA_TOOLKIT_ROOT_DIR: "
        "${CUDA_TOOLKIT_ROOT_DIR}, please check if the path is correct.")
endif ()

set(__TENSORRT_LIB_COMPONENTS nvinfer;nvinfer_plugin)
foreach(__component ${__TENSORRT_LIB_COMPONENTS})
    find_library(
        __component_path ${__component}
        HINTS ${TENSORRT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
    if (NOT __component_path)
        message(FATAL_ERROR "Cannot find TensorRT lib ${__component} in "
            "TENSORRT_DIR: ${TENSORRT_DIR} or CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}, "
            "please check if the path is correct")
    endif()

    add_library(${__component} SHARED IMPORTED)
    set_property(TARGET ${__component} APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    if (MSVC)
        set_target_properties(
            ${__component} PROPERTIES
            IMPORTED_IMPLIB_RELEASE ${__component_path}
            INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR}
        )
    else()
        set_target_properties(
            ${__component} PROPERTIES
            IMPORTED_LOCATION_RELEASE ${__component_path}
            INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR}
        )
    endif()
    unset(__component_path CACHE)
endforeach()

set(TENSORRT_LIBS ${__TENSORRT_LIB_COMPONENTS})
