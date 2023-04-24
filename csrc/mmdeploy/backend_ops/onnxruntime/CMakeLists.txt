# Copyright (c) OpenMMLab. All rights reserved.

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "OFF")
endif()

if (MSVC)
    #Always enable exception handling, even for Windows ARM
    string(APPEND CMAKE_CXX_FLAGS " /EHsc")
    string(APPEND CMAKE_C_FLAGS " /EHsc")

    # AVX2
    string(APPEND CMAKE_CXX_FLAGS " /arch:AVX2")
    string(APPEND CMAKE_C_FLAGS " /arch:AVX2")
else()
    # AVX2
    string(APPEND CMAKE_CXX_FLAGS " -mavx2")
    string(APPEND CMAKE_C_FLAGS " -mavx2")
endif()

project(mmdeploy_onnxruntime_ops)

if (!APPLE)
    include(${CMAKE_SOURCE_DIR}/cmake/cuda.cmake)
endif()
include(${CMAKE_SOURCE_DIR}/cmake/modules/FindONNXRUNTIME.cmake)

# add plugin source

file(GLOB_RECURSE ORT_OPS_SRCS *.cpp *.cu)
if (NOT FOUND_CUDA)
    set (EXCLUDE_CUDA_DIR "/cuda/")
    foreach (TMP_PATH ${ORT_OPS_SRCS})
        string (FIND ${TMP_PATH} ${EXCLUDE_CUDA_DIR} EXCLUDE_DIR_FOUND)
        if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
            list (REMOVE_ITEM ORT_OPS_SRCS ${TMP_PATH})
        endif ()
    endforeach(TMP_PATH)
endif()
add_library(${PROJECT_NAME}_obj OBJECT "${ORT_OPS_SRCS}")
target_compile_definitions(${PROJECT_NAME}_obj PRIVATE -DMMDEPLOY_API_EXPORTS=1)
target_compile_options(${PROJECT_NAME}_obj PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)
set_target_properties(${PROJECT_NAME}_obj PROPERTIES POSITION_INDEPENDENT_CODE 1)
mmdeploy_export(${PROJECT_NAME}_obj)

target_include_directories(${PROJECT_NAME}_obj
        PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)

target_include_directories(${PROJECT_NAME}_obj PUBLIC
        $<BUILD_INTERFACE:${ONNXRUNTIME_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/csrc>)
if (FOUND_CUDA)
    target_link_libraries(${PROJECT_NAME}_obj PUBLIC onnxruntime cublas cudart)
else()
    target_link_libraries(${PROJECT_NAME}_obj PUBLIC onnxruntime)
endif()

mmdeploy_add_library(${PROJECT_NAME} SHARED EXCLUDE "")
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}_obj)
if (APPLE)
        set_target_properties(${PROJECT_NAME} PROPERTIES
                BUILD_RPATH "@loader_path"
                INSTALL_RPATH "@loader_path")
else ()
        set_target_properties(${PROJECT_NAME} PROPERTIES
                BUILD_RPATH "\$ORIGIN"
                INSTALL_RPATH "\$ORIGIN")
endif ()
add_library(mmdeploy::onnxruntime::ops ALIAS ${PROJECT_NAME})

set(_ORT_OPS_DIR ${CMAKE_SOURCE_DIR}/mmdeploy/lib)
install(TARGETS ${PROJECT_NAME} DESTINATION ${_ORT_OPS_DIR})
