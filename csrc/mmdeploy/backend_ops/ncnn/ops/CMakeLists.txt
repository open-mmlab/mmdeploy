# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_ncnn_ops)

# add plugin source
file(GLOB_RECURSE NCNN_OPS_SRCS *.cpp)
add_library(${PROJECT_NAME}_obj OBJECT "${NCNN_OPS_SRCS}")
target_compile_definitions(${PROJECT_NAME}_obj PRIVATE -DMMDEPLOY_API_EXPORTS=1)
set_target_properties(${PROJECT_NAME}_obj PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_link_libraries(${PROJECT_NAME}_obj PRIVATE ncnn)
set(_COMMON_INCLUDE_DIRS
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/csrc>)
target_include_directories(${PROJECT_NAME}_obj
        PUBLIC ${_COMMON_INCLUDE_DIRS})
mmdeploy_export(${PROJECT_NAME}_obj)

mmdeploy_add_library(${PROJECT_NAME} SHARED EXCLUDE "")
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_obj)
target_include_directories(${PROJECT_NAME}
        PUBLIC ${_COMMON_INCLUDE_DIRS})

add_library(mmdeploy::ncnn_ops ALIAS ${PROJECT_NAME})

set(_NCNN_OPS_DIR ${CMAKE_SOURCE_DIR}/mmdeploy/lib)
install(TARGETS ${PROJECT_NAME} DESTINATION ${_NCNN_OPS_DIR})
