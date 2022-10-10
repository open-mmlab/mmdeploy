# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_mmaction_cpu_impl CXX)

if ("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES)
    add_library(${PROJECT_NAME} OBJECT format_shape_impl.cpp)
    set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE 1)
    if (NOT (MMDEPLOY_SHARED_LIBS OR MSVC))
        target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)
    endif ()
    target_link_libraries(${PROJECT_NAME} PRIVATE
            mmdeploy::core)
    target_link_libraries(mmdeploy_mmaction PRIVATE ${PROJECT_NAME})
    mmdeploy_export(${PROJECT_NAME})
endif ()
