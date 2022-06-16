# Copyright (c) OpenMMLab. All rights reserved.

project(ts_optimizer)

find_package(Torch REQUIRED)
if (NOT TARGET pybind11)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 pybind11)
endif ()

file(GLOB_RECURSE OPTIMIZER_SRCS *.cpp)

pybind11_add_module(${PROJECT_NAME} ${OPTIMIZER_SRCS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
target_link_directories(${PROJECT_NAME} PRIVATE mmdeploy::torchscript_ops)
set_target_properties(
        ${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_SOURCE_DIR}/mmdeploy/backend/torchscript)
