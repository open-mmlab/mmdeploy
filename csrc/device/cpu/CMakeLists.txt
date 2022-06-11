# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_cpu_device)

file(GLOB_RECURSE SRCS "*.cpp")

mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

add_library(mmdeploy::device::cpu ALIAS ${PROJECT_NAME})
