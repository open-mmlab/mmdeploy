# Copyright (c) OpenMMLab. All rights reserved.
if (NOT "cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
    return()
endif ()

project(mmdeploy_mmaction_cuda_impl CXX)

add_library(${PROJECT_NAME} OBJECT format_shape_impl.cpp transpose.cu)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE 1)
if (NOT (MMDEPLOY_SHARED_LIBS OR MSVC))
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)
endif ()
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE
    mmdeploy::core)
target_link_libraries(mmdeploy_mmaction PRIVATE ${PROJECT_NAME})
mmdeploy_export(${PROJECT_NAME})
