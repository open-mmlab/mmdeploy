# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_rknn_net)

mmdeploy_add_module(${PROJECT_NAME} rknn_net.cpp)

add_library(rknn SHARED IMPORTED)
if(DEFINED ENV{RKNPU2_DEVICE_DIR} OR DEFINED RKNPU2_DEVICE_DIR)
    set_target_properties(rknn PROPERTIES
            IMPORTED_LOCATION "${RKNPU2_DEVICE_DIR}/Linux/librknn_api/aarch64/librknn_api.so"
            INTERFACE_INCLUDE_DIRECTORIES "${RKNPU2_DEVICE_DIR}/Linux/librknn_api/include"
            )
    target_compile_definitions(${PROJECT_NAME} PRIVATE RK_MODELS)
elseif(DEFINED ENV{RKNPU_RUNTIME_DIR} OR DEFINED RKNPU_RUNTIME_DIR)
    set_target_properties(rknn PROPERTIES IMPORTED_CONFIGURATIONS RELEASE
            IMPORTED_LOCATION_RELEASE "${RKNPU_RUNTIME_DIR}/lib/librknn_api.so"
            INTERFACE_INCLUDE_DIRECTORIES "${RKNPU_RUNTIME_DIR}/include"
            )
    target_compile_definitions(${PROJECT_NAME} PRIVATE RV_MODELS)
else()
    message(FATAL_ERROR "RKNPU2_DEVICE_DIR env must be defined")
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE rknn)
add_library(mmdeploy::rknn_net ALIAS ${PROJECT_NAME})
