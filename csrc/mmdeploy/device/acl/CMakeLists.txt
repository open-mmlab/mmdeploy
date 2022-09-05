# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_acl_device)

file(GLOB_RECURSE SRCS "*.cpp")

mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
