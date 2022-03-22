# Copyright (c) OpenMMLab. All rights reserved.

if (NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
endif ()
if (NOT ONNXRUNTIME_DIR)
    message(FATAL_ERROR "Please set ONNXRUNTIME_DIR with cmake -D option.")
endif()

find_path(
    ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_DIR}
    PATH_SUFFIXES include)
find_library(
    ONNXRUNTIME_LIBRARY_ONNXRUNTIME_PATH onnxruntime
    HINTS ${ONNXRUNTIME_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
if (NOT (ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY_ONNXRUNTIME_PATH))
    message(FATAL_ERROR "Couldn't find onnxruntime in ONNXRUNTIME_DIR: "
        "${ONNXRUNTIME_DIR}, please check if the path is correct.")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_property(TARGET onnxruntime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
if (MSVC)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_IMPLIB_RELEASE ${ONNXRUNTIME_LIBRARY_ONNXRUNTIME_PATH}
        INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_INCLUDE_DIR}
    )

else()
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION_RELEASE ${ONNXRUNTIME_LIBRARY_ONNXRUNTIME_PATH}
        INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_INCLUDE_DIR}
    )
endif()
