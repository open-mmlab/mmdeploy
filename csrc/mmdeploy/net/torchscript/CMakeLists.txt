# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_torch_net)

find_package(Torch REQUIRED)
find_package(TorchVision)

mmdeploy_add_module(${PROJECT_NAME} torch_net.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE
        #        mmdeploy_torchscript_ops
        ${TORCH_LIBRARIES})
if (TorchVision_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE
            -Wl,--no-as-needed
            TorchVision::TorchVision
            -Wl,--as-needed)
endif ()

add_library(mmdeploy::torch_net ALIAS ${PROJECT_NAME})
