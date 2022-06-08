# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_mmocr_cpu_impl CXX)

if ("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES)
    add_library(${PROJECT_NAME} OBJECT dbnet.cpp panet.cpp psenet.cpp)
    set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE 1)
    target_link_libraries(${PROJECT_NAME} PRIVATE
            mmdeploy_opencv_utils
            mmdeploy::core)
    target_link_libraries(mmdeploy_mmocr PRIVATE ${PROJECT_NAME})
    mmdeploy_export(${PROJECT_NAME})
endif ()
