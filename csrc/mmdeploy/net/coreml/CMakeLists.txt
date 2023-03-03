# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_coreml_net)

if ("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES)
    find_library(CORE_ML CoreML)
    find_library(FOUNDATION Foundation)
    mmdeploy_add_net(${PROJECT_NAME} coreml_net.mm)
    target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CORE_ML} ${FOUNDATION})
    add_library(mmdeploy::coreml_net ALIAS ${PROJECT_NAME})
else ()
    message(ERROR "'coreml_net' is NOT supported in target devices: ${MMDEPLOY_TARGET_DEVICES}")
endif ()
