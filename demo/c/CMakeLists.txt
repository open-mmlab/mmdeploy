# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_capi_demo)

if (NOT (${CMAKE_PROJECT_NAME} STREQUAL "MMDeploy"))
    find_package(MMDeploy REQUIRED)
endif ()

set(TASK_LIST ${MMDEPLOY_TASKS})

if ("text_detector" IN_LIST TASK_LIST OR "text_recognizer" IN_LIST TASK_LIST)
    list(REMOVE_ITEM TASK_LIST "text_detector" "text_recognizer")
    add_subdirectory(ocr)
endif()

foreach(TASK ${TASK_LIST})
    add_subdirectory(${TASK})
endforeach()
