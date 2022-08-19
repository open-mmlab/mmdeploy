# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.14)
project(mmdeploy_cxx_api)

if (MMDEPLOY_BUILD_SDK_CXX_API)
    add_library(${PROJECT_NAME} INTERFACE)
    target_include_directories(${PROJECT_NAME} INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>)
    target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
    target_link_libraries(${PROJECT_NAME} INTERFACE mmdeploy::core)
    foreach (task ${MMDEPLOY_TASKS})
        target_link_libraries(mmdeploy_${task} INTERFACE ${PROJECT_NAME})
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/mmdeploy/${task}.hpp
                DESTINATION include/mmdeploy)
    endforeach ()
    if (TARGET mmdeploy)
        target_link_libraries(mmdeploy INTERFACE ${PROJECT_NAME})
    endif ()
    mmdeploy_export(${PROJECT_NAME})
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/mmdeploy/common.hpp
            DESTINATION include/mmdeploy)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/demo/csrc/ DESTINATION example/cpp
            FILES_MATCHING
            PATTERN "*.cxx"
            )
endif ()
