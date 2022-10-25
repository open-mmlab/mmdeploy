# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_monitor_utils)

# include(${CMAKE_SOURCE_DIR}/cmake/cuda.cmake NO_POLICY_SCOPE)

add_executable(monitor monitor.cpp resource_monitor.cpp)
target_include_directories(monitor PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(monitor mmdeploy::core)
if ("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
    target_compile_definitions(monitor PRIVATE USE_CUDA)
    target_link_libraries(monitor ${CUDA_LIBRARIES} nvidia-ml) # nvml
endif ()
