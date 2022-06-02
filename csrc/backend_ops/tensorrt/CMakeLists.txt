# Copyright (c) OpenMMLab. All rights reserved.
project(mmdeploy_tensorrt_ops)

include(${CMAKE_SOURCE_DIR}/cmake/tensorrt.cmake)

# cub
if (NOT DEFINED CUB_ROOT_DIR)
    if (CUDA_VERSION VERSION_LESS 11.0)
        set(CUB_ROOT_DIR "${CMAKE_SOURCE_DIR}/third_party/cub")
    endif ()
endif ()

file(GLOB_RECURSE BACKEND_OPS_SRCS *.cpp *.cu)
add_library(${PROJECT_NAME}_obj OBJECT "${BACKEND_OPS_SRCS}")
set_target_properties(${PROJECT_NAME}_obj PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_compile_definitions(${PROJECT_NAME}_obj
        PRIVATE -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT=1)
target_include_directories(${PROJECT_NAME}_obj
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_include_directories(${PROJECT_NAME}_obj
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_include_directories(${PROJECT_NAME}_obj
        PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_include_directories(${PROJECT_NAME}_obj PRIVATE ${TENSORRT_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME}_obj PRIVATE ${CUDNN_DIR}/include)
target_include_directories(${PROJECT_NAME}_obj PRIVATE ${CUB_ROOT_DIR})
target_link_libraries(${PROJECT_NAME}_obj
        PUBLIC ${TENSORRT_LIBS} cublas cudnn)
mmdeploy_export(${PROJECT_NAME}_obj)

# Build module library. It is used to convert onnx model to tensorrt engine
mmdeploy_add_module(${PROJECT_NAME} MODULE EXCLUDE "")
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_obj)
add_library(mmdeploy::tensorrt_ops ALIAS ${PROJECT_NAME})

set(_TRT_OPS_DIR ${CMAKE_SOURCE_DIR}/mmdeploy/lib)
install(TARGETS ${PROJECT_NAME} DESTINATION ${_TRT_OPS_DIR})
