# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_ipu_net)

find_package(model_runtime REQUIRED)
mmdeploy_add_module(${PROJECT_NAME} ipu_net.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
        ${MODEL_RUNTIME_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE model_runtime)

add_library(mmdeploy::ipu_net ALIAS ${PROJECT_NAME})
