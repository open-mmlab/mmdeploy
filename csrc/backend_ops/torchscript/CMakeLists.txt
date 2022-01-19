# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
include(${CMAKE_SOURCE_DIR}/cmake/cuda.cmake NO_POLICY_SCOPE)

if("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
    project(mmdeploy_torchscript_ops CUDA CXX)
    include(${CMAKE_SOURCE_DIR}/cmake/cuda.cmake NO_POLICY_SCOPE)
    file(GLOB_RECURSE BACKEND_OPS_SRCS *.cpp *.cu)
else()
    project(mmdeploy_torchscript_ops CXX)
    file(GLOB_RECURSE BACKEND_OPS_SRCS *.cpp)
endif()

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
find_package(Torch REQUIRED)

set_targets(${PROJECT_NAME} BACKEND_OPS_OBJ BACKEND_OPS_STATIC BACKEND_OPS_MODULE)

build_object_target(${BACKEND_OPS_OBJ} "${BACKEND_OPS_SRCS}")
target_compile_definitions(${BACKEND_OPS_OBJ}
        PRIVATE -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT=1)
target_include_directories(${BACKEND_OPS_OBJ}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_include_directories(${BACKEND_OPS_OBJ}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)

if("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
target_include_directories(${BACKEND_OPS_OBJ}
        PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
endif()
target_link_libraries(${BACKEND_OPS_OBJ} PRIVATE ${TORCH_LIBRARIES})

# Build module library. It is used to inference with torchscript
build_module_target(${BACKEND_OPS_MODULE} ${BACKEND_OPS_OBJ} "PRIVATE")
add_library(mmdeploy::torchscript_ops ALIAS ${BACKEND_OPS_MODULE})
install_targets(${BACKEND_OPS_MODULE})

if (MMDEPLOY_BUILD_SDK)
    ## Build static library. SDK's uses it to build `trt_net` module
    build_static_target(${BACKEND_OPS_STATIC} ${BACKEND_OPS_OBJ} "PRIVATE")
    add_library(mmdeploy::torchscript_ops::static ALIAS ${BACKEND_OPS_STATIC})
endif ()
