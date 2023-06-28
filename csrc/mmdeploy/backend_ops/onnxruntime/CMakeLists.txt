# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_onnxruntime_ops)

include(${CMAKE_SOURCE_DIR}/cmake/modules/FindONNXRUNTIME.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)

# add plugin source
file(GLOB_RECURSE ORT_OPS_SRCS *.cpp)
add_library(${PROJECT_NAME}_obj OBJECT "${ORT_OPS_SRCS}")
target_compile_definitions(${PROJECT_NAME}_obj PRIVATE -DMMDEPLOY_API_EXPORTS=1)
target_compile_options(${PROJECT_NAME}_obj PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)
set_target_properties(${PROJECT_NAME}_obj PROPERTIES POSITION_INDEPENDENT_CODE 1)
mmdeploy_export(${PROJECT_NAME}_obj)

target_include_directories(${PROJECT_NAME}_obj PUBLIC
        $<BUILD_INTERFACE:${ONNXRUNTIME_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/csrc>)
target_link_libraries(${PROJECT_NAME}_obj PUBLIC onnxruntime)

mmdeploy_add_library(${PROJECT_NAME} SHARED EXCLUDE "")
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}_obj)
mmdeploy_add_rpath(${PROJECT_NAME})
add_library(mmdeploy::onnxruntime::ops ALIAS ${PROJECT_NAME})

set(_ORT_OPS_DIR ${CMAKE_SOURCE_DIR}/mmdeploy/lib)
install(TARGETS ${PROJECT_NAME} DESTINATION ${_ORT_OPS_DIR})
