# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_acl_net)

if ("acl" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    message(STATUS "${ASCEND_TOOLKIT_HOME}")
    mmdeploy_add_module(${PROJECT_NAME} acl_net.cpp)
    target_include_directories(${PROJECT_NAME} PRIVATE 
        $<BUILD_INTERFACE:${ASCEND_TOOLKIT_HOME}/acllib/include>)
    target_link_directories(${PROJECT_NAME} PUBLIC 
        # ${ASCEND_TOOLKIT_HOME}/runtime/lib64/stub
        ${ASCEND_TOOLKIT_HOME}/lib64
        ${ASCEND_TOOLKIT_HOME}/../../driver/lib64)
    target_link_libraries(${PROJECT_NAME} PRIVATE ascendcl)
endif ()