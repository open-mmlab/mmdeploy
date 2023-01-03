# Copyright (c) OpenMMLab. All rights reserved.

if("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
    project(mmdeploy_torchscript_ops CUDA CXX)
    file(GLOB_RECURSE BACKEND_OPS_SRCS *.cpp *.cu)
else()
    project(mmdeploy_torchscript_ops CXX)
    file(GLOB_RECURSE BACKEND_OPS_SRCS *.cpp)
endif()

find_package(Torch REQUIRED)


add_library(${PROJECT_NAME}_obj OBJECT "${BACKEND_OPS_SRCS}")
set_target_properties(${PROJECT_NAME}_obj PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_compile_definitions(${PROJECT_NAME}_obj
    PRIVATE -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT=1)
target_include_directories(${PROJECT_NAME}_obj
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
target_include_directories(${PROJECT_NAME}_obj
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)

if("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
    target_include_directories(${PROJECT_NAME}_obj
        PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
endif()
target_link_libraries(${PROJECT_NAME}_obj PRIVATE ${TORCH_LIBRARIES})
mmdeploy_export(${PROJECT_NAME}_obj)

# Build module library. It is used to inference with torchscript
mmdeploy_add_module(${PROJECT_NAME} MODULE EXCLUDE "")
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}_obj)
add_library(mmdeploy::torchscript_ops ALIAS ${PROJECT_NAME})

set(_TORCHJIT_OPS_DIR ${CMAKE_SOURCE_DIR}/mmdeploy/lib)
install(TARGETS ${PROJECT_NAME} DESTINATION ${_TORCHJIT_OPS_DIR})
