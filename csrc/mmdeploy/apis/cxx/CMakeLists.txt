# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.14)
project(mmdeploy_cxx_api)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
set(_tasks ${MMDEPLOY_TASKS} pipeline)
foreach (task ${_tasks})
    target_link_libraries(mmdeploy_${task} INTERFACE ${PROJECT_NAME})
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/mmdeploy/${task}.hpp
            DESTINATION include/mmdeploy)
endforeach ()
if (TARGET mmdeploy)
    target_include_directories(${PROJECT_NAME} INTERFACE
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/csrc>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/outcome>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/json>
            )
    target_include_directories(${PROJECT_NAME} INTERFACE
            $<INSTALL_INTERFACE:include>
            $<INSTALL_INTERFACE:include/mmdeploy/third_party/outcome>
            $<INSTALL_INTERFACE:include/mmdeploy/third_party/json>
            )
    if (NOT MMDEPLOY_SPDLOG_EXTERNAL)
        target_include_directories(${PROJECT_NAME} INTERFACE
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/spdlog/include>
                $<INSTALL_INTERFACE:include/mmdeploy/third_party>)
    endif ()
    target_link_libraries(mmdeploy INTERFACE ${PROJECT_NAME})
else ()
    target_link_libraries(${PROJECT_NAME} INTERFACE mmdeploy::core)
endif ()
mmdeploy_export(${PROJECT_NAME})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/mmdeploy/common.hpp
        DESTINATION include/mmdeploy)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/demo/csrc/ DESTINATION example/cpp
        FILES_MATCHING
        PATTERN "*.cxx"
        PATTERN "*.h"
        )
