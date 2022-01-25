# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_onnxruntime_ops)

include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)

# add plugin source
file(GLOB_RECURSE ORT_OPS_SRCS *.cpp)
add_library(${PROJECT_NAME}_obj OBJECT "${ORT_OPS_SRCS}")
target_compile_definitions(${PROJECT_NAME}_obj PRIVATE -DMMDEPLOY_API_EXPORTS=1)
set_target_properties(${PROJECT_NAME}_obj PROPERTIES POSITION_INDEPENDENT_CODE 1)

target_include_directories(${PROJECT_NAME}_obj PUBLIC
        $<BUILD_INTERFACE:${ONNXRUNTIME_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>)
target_link_directories(${PROJECT_NAME}_obj PUBLIC
        ${ONNXRUNTIME_DIR}/lib)
target_link_libraries(${PROJECT_NAME}_obj PUBLIC onnxruntime)

mmdeploy_add_library(${PROJECT_NAME} SHARED $<TARGET_OBJECTS:${ORT_OPS_OBJ}>)
target_link_libraries(${PROJECT_NAME} PUBLIC ${ORT_OPS_OBJ})
add_library(mmdeploy::onnxruntime::ops ALIAS ${PROJECT_NAME})
mmdeploy_export(${ORT_OPS_OBJ})

if (MMDEPLOY_BUILD_SDK)
    if (BUILD_SHARED_LIBS)
        target_link_libraries(MMDeployDynamicModules INTERFACE ${ORT_OPS_MODULE})
    else ()
        mmdeploy_add_module(${PROJECT_NAME}_static "")
        target_link_libraries(${PROJECT_NAME}_static PRIVATE ${PROJECT_NAME}_obj)
    endif ()
endif ()
