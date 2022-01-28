# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(ts_optimizer)

find_package(Torch REQUIRED)

file(GLOB_RECURSE OPTIMIZER_SRCS *.cpp)

add_executable(${PROJECT_NAME} ${OPTIMIZER_SRCS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
target_link_directories(${PROJECT_NAME} PRIVATE mmdeploy::torchscript_ops)
