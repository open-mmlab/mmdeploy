# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_codebase)

set(CODEBASES "")
if ("all" IN_LIST MMDEPLOY_CODEBASES)
    list(APPEND CODEBASES "mmcls")
    list(APPEND CODEBASES "mmdet")
    list(APPEND CODEBASES "mmseg")
    list(APPEND CODEBASES "mmocr")
    list(APPEND CODEBASES "mmedit")
    list(APPEND CODEBASES "mmpose")
    list(APPEND CODEBASES "mmrotate")
else ()
    set(CODEBASES ${MMDEPLOY_CODEBASES})
endif ()

foreach (codebase IN LISTS CODEBASES)
    message(STATUS "build codebase: ${codebase}")
    add_subdirectory(${codebase})
endforeach ()
