# Copyright (c) OpenMMLab. All rights reserved.

project(capis)

set(CAPI_OBJS)

macro(add_object name)
    add_library(${name} OBJECT ${ARGN})
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE 1)
    target_compile_definitions(${name} PRIVATE -DMMDEPLOY_API_EXPORTS=1)
    if (NOT MSVC)
        target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)
    endif ()
    target_link_libraries(${name} PRIVATE mmdeploy::core)
    set(CAPI_OBJS ${CAPI_OBJS} ${name})
    mmdeploy_export(${name})
endmacro()

set(COMMON_LIST
        common
        model
        executor
        pipeline)

set(TASK_LIST ${MMDEPLOY_TASKS})

foreach (TASK ${COMMON_LIST})
    set(TARGET_NAME mmdeploy_${TASK})
    set(OBJECT_NAME mmdeploy_${TASK}_obj)
    add_object(${OBJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/mmdeploy/${TASK}.cpp)
    mmdeploy_add_library(${TARGET_NAME})
    target_link_libraries(${TARGET_NAME} PRIVATE ${OBJECT_NAME})
    target_include_directories(${TARGET_NAME} PUBLIC
            $<INSTALL_INTERFACE:include>)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/mmdeploy/${TASK}.h
            DESTINATION include/mmdeploy)
endforeach ()

target_link_libraries(mmdeploy_executor PUBLIC
        mmdeploy_execution mmdeploy_common)
target_link_libraries(mmdeploy_pipeline PUBLIC
        mmdeploy_executor mmdeploy_model mmdeploy_common)

foreach (TASK ${TASK_LIST})
    set(TARGET_NAME mmdeploy_${TASK})
    set(OBJECT_NAME mmdeploy_${TASK}_obj)
    add_object(${OBJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/mmdeploy/${TASK}.cpp)
    mmdeploy_add_library(${TARGET_NAME})
    target_link_libraries(${TARGET_NAME} PRIVATE ${OBJECT_NAME}
            mmdeploy_pipeline)
    target_include_directories(${TARGET_NAME} PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/mmdeploy/${TASK}.h
            DESTINATION include/mmdeploy)
endforeach ()

install(DIRECTORY ${CMAKE_SOURCE_DIR}/demo/csrc/ DESTINATION example/cpp
        FILES_MATCHING
        PATTERN "*.cpp"
        PATTERN "CMakeLists.txt"
        )

if (MMDEPLOY_BUILD_SDK_CSHARP_API OR MMDEPLOY_BUILD_SDK_MONOLITHIC)
    add_library(mmdeploy SHARED)
    mmdeploy_load_static(mmdeploy MMDeployStaticModules)
    mmdeploy_load_dynamic(mmdeploy MMDeployDynamicModules)
    target_link_libraries(mmdeploy PRIVATE ${CAPI_OBJS} mmdeploy_execution)
    target_include_directories(mmdeploy PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>)
    set(MMDEPLOY_VERSION ${MMDEPLOY_VERSION_MAJOR}
            .${MMDEPLOY_VERSION_MINOR}
            .${MMDEPLOY_VERSION_PATCH})
    string(REPLACE ";" "" MMDEPLOY_VERSION ${MMDEPLOY_VERSION})
    set_target_properties(mmdeploy PROPERTIES
            VERSION ${MMDEPLOY_VERSION}
            SOVERSION ${MMDEPLOY_VERSION_MAJOR})
    mmdeploy_export(mmdeploy)
endif ()
