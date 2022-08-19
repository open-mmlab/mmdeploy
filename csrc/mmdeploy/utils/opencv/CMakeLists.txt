# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_opencv_utils)

mmdeploy_add_library(${PROJECT_NAME} opencv_utils.cpp)

target_link_libraries(${PROJECT_NAME}
        PRIVATE mmdeploy::core
        PUBLIC ${OpenCV_LIBS})

target_include_directories(${PROJECT_NAME}
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
