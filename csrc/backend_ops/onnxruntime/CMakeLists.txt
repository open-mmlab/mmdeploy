# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_onnxruntime_ops)

include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)
#set_targets(${PROJECT_NAME} ORT_OPS_OBJ ORT_OPS_STATIC ORT_OPS_MODULE)

# add plugin source
file(GLOB_RECURSE ORT_OPS_SRCS *.cpp)
#build_object_target(${ORT_OPS_OBJ} "${ORT_OPS_SRCS}")
set(ORT_OPS_OBJ ${PROJECT_NAME}_obj)
add_library(${ORT_OPS_OBJ} OBJECT "${ORT_OPS_SRCS}")
target_compile_definitions(${ORT_OPS_OBJ} PRIVATE -DMMDEPLOY_API_EXPORTS=1)
set_target_properties(${ORT_OPS_OBJ} PROPERTIES POSITION_INDEPENDENT_CODE 1)

target_include_directories(${ORT_OPS_OBJ} PUBLIC
        $<BUILD_INTERFACE:${ONNXRUNTIME_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>)
target_link_directories(${ORT_OPS_OBJ} PUBLIC
        ${ONNXRUNTIME_DIR}/lib)
target_link_libraries(${ORT_OPS_OBJ} PUBLIC onnxruntime)

set(ORT_OPS_MODULE ${PROJECT_NAME})
mmdeploy_add_library(${ORT_OPS_MODULE} SHARED $<TARGET_OBJECTS:${ORT_OPS_OBJ}>)
target_link_libraries(${ORT_OPS_MODULE} PUBLIC ${ORT_OPS_OBJ})
add_library(mmdeploy::onnxruntime::ops ALIAS ${ORT_OPS_MODULE})
mmdeploy_export(${ORT_OPS_OBJ})

#install_targets(${ORT_OPS_MODULE})
if (MMDEPLOY_BUILD_SDK)
    ## Build static library. SDK's uses it to build `ort_net` module
    if (BUILD_SHARED_LIBS)
        target_link_libraries(MMDeployDynamicModules INTERFACE ${ORT_OPS_MODULE})
#        mmdeploy_export(${ORT_OPS_MODULE})
    else ()
        set(ORT_OPS_STATIC ${PROJECT_NAME}_static)
        mmdeploy_add_module(${ORT_OPS_STATIC} "")
        target_link_libraries(${ORT_OPS_STATIC} PRIVATE ${ORT_OPS_OBJ})
    endif ()
endif ()
