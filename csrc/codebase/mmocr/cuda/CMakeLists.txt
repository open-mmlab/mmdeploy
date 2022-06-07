# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_mmocr_cuda_impl)

if ("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
    add_library(${PROJECT_NAME} OBJECT
            connected_component.cu
            utils.cu
            dbnet.cpp
            panet.cpp
            psenet.cpp)
    set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE 1)
    target_include_directories(${PROJECT_NAME} PRIVATE
            ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE
            mmdeploy_opencv_utils
            mmdeploy::core
            ${CUDA_LIBRARIES}
            cuda)
    target_link_libraries(mmdeploy_mmocr PRIVATE
            ${PROJECT_NAME})
    mmdeploy_export(${PROJECT_NAME})
endif ()
