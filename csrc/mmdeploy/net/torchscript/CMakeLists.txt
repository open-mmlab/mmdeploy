# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_torch_net)

option(MMDEPLOY_TORCHSCRIPT_SDK_BACKEND "Build TorchScript SDK backend" OFF)

if (MMDEPLOY_TORCHSCRIPT_SDK_BACKEND)
    find_package(Torch REQUIRED)
    find_package(TorchVision QUIET)

    mmdeploy_add_module(${PROJECT_NAME} torch_net.cpp)

    target_link_libraries(${PROJECT_NAME} PRIVATE
            ${TORCH_LIBRARIES})

    target_link_directories(${PROJECT_NAME} INTERFACE
            $<BUILD_INTERFACE:${Torch_DIR}/../../../lib>)

    target_link_libraries(${PROJECT_NAME} PRIVATE
            mmdeploy_torchscript_ops_obj)

    if (TorchVision_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE TorchVision::TorchVision)
        target_compile_definitions(${PROJECT_NAME} PRIVATE -DMMDEPLOY_USE_TORCHVISION=1)
    endif ()

    add_library(mmdeploy::torch_net ALIAS ${PROJECT_NAME})
endif ()
