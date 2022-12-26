# Copyright (c) OpenMMLab. All rights reserved.

if (NOT DEFINED TVM_DIR)
    set(TVM_DIR $ENV{TVM_DIR})
endif ()
if (NOT TVM_DIR)
    message(FATAL_ERROR "Please set TVM_DIR with cmake -D option.")
endif()

find_path(
    TVM_INCLUDE_DIR tvm/runtime/c_runtime_api.h
    HINTS ${TVM_DIR}
    PATH_SUFFIXES include)

find_path(
    DMLC_CORE_INCLUDE_DIR  dmlc/io.h
    HINTS ${TVM_DIR}/3rdparty/dmlc-core
    PATH_SUFFIXES include)

find_path(
    DLPACK_INCLUDE_DIR dlpack/dlpack.h
    HINTS ${TVM_DIR}/3rdparty/dlpack
    PATH_SUFFIXES include)

find_library(
    TVM_LIBRARY_PATH tvm_runtime
    HINTS ${TVM_DIR}
    PATH_SUFFIXES build lib build/${CMAKE_BUILD_TYPE})
if (NOT (TVM_INCLUDE_DIR AND DMLC_CORE_INCLUDE_DIR AND DLPACK_INCLUDE_DIR AND TVM_LIBRARY_PATH))
    message(FATAL_ERROR "Couldn't find tvm in TVM_DIR: "
        "${TVM_DIR}, please check if the path is correct.")
endif()

add_library(tvm_runtime SHARED IMPORTED)
set_property(TARGET tvm_runtime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
if (MSVC)
    set_target_properties(tvm_runtime PROPERTIES
        IMPORTED_IMPLIB_RELEASE ${TVM_LIBRARY_PATH}
        INTERFACE_INCLUDE_DIRECTORIES ${TVM_INCLUDE_DIR} ${DMLC_CORE_INCLUDE_DIR} ${DLPACK_INCLUDE_DIR}
    )

else()
    set_target_properties(tvm_runtime PROPERTIES
        IMPORTED_LOCATION_RELEASE ${TVM_LIBRARY_PATH}
        INTERFACE_INCLUDE_DIRECTORIES ${TVM_INCLUDE_DIR} ${DMLC_CORE_INCLUDE_DIR} ${DLPACK_INCLUDE_DIR}
    )
endif()
