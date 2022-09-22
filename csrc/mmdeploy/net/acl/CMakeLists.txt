# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_acl_net)

if ("acl" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    if (NOT DEFINED ASCEND_TOOLKIT_HOME)
        set(ASCEND_TOOLKIT_HOME $ENV{ASCEND_TOOLKIT_HOME})
    endif ()
    mmdeploy_add_module(${PROJECT_NAME} acl_net.cpp)
    target_include_directories(${PROJECT_NAME} PRIVATE
        $<BUILD_INTERFACE:${ASCEND_TOOLKIT_HOME}/runtime/include>)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        $<BUILD_INTERFACE:${ASCEND_TOOLKIT_HOME}/runtime/lib64/stub/libascendcl.so>)
endif ()
