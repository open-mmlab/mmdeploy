# Copyright (c) OpenMMLab. All rights reserved.

project(capis)

set(COMMON_LIST
        common
        model
        executor
        pipeline)

set(TASK_LIST)

if ("all" IN_LIST MMDEPLOY_CODEBASES)
    list(APPEND TASK_LIST
            classifier
            detector
            segmentor
            text_detector
            text_recognizer
            pose_detector
            restorer
            rotated_detector)
else ()
    if ("mmcls" IN_LIST MMDEPLOY_CODEBASES)
        list(APPEND TASK_LIST classifier)
    endif ()
    if ("mmdet" IN_LIST MMDEPLOY_CODEBASES)
        list(APPEND TASK_LIST detector)
    endif ()
    if ("mmseg" IN_LIST MMDEPLOY_CODEBASES)
        list(APPEND TASK_LIST segmentor)
    endif ()
    if ("mmedit" IN_LIST MMDEPLOY_CODEBASES)
        list(APPEND TASK_LIST restorer)
    endif ()
    if ("mmocr" IN_LIST MMDEPLOY_CODEBASES)
        list(APPEND TASK_LIST text_detector)
        list(APPEND TASK_LIST text_recognizer)
    endif ()
    if ("mmpose" IN_LIST MMDEPLOY_CODEBASES)
        list(APPEND TASK_LIST pose_detector)
    endif ()
    if ("mmrotate" IN_LIST MMDEPLOY_CODEBASES)
        list(APPEND TASK_LIST "rotated_detector")
    endif ()
endif ()

foreach (TASK ${COMMON_LIST})
    set(TARGET_NAME mmdeploy_${TASK})
    mmdeploy_add_library(${TARGET_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${TASK}.cpp)
    target_link_libraries(${TARGET_NAME} PRIVATE mmdeploy::core)
    target_include_directories(${TARGET_NAME} PUBLIC
            $<INSTALL_INTERFACE:include/mmdeploy/apis/c>)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${TASK}.h
            DESTINATION include/mmdeploy/apis/c)
endforeach ()

target_link_libraries(mmdeploy_executor PUBLIC
        mmdeploy_common)
target_link_libraries(mmdeploy_pipeline PUBLIC
        mmdeploy_executor mmdeploy_model mmdeploy_common)

foreach (TASK ${TASK_LIST})
    set(TARGET_NAME mmdeploy_${TASK})
    mmdeploy_add_library(${TARGET_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${TASK}.cpp)

    target_link_libraries(${TARGET_NAME} PRIVATE
            mmdeploy_pipeline mmdeploy::core)
    target_include_directories(${TARGET_NAME} PUBLIC
            $<INSTALL_INTERFACE:include/mmdeploy/apis/c>)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${TASK}.h
            DESTINATION include/mmdeploy/apis/c)
endforeach ()

if (MMDEPLOY_BUILD_SDK_CSHARP_API)
    # build MMDeployExtern.dll just for csharp nuget package.
    # no Installation for c/c++ package.
    file(GLOB SRCS "*.c" "*.cpp")
    add_library(MMDeployExtern SHARED ${SRCS})
    target_compile_definitions(MMDeployExtern PRIVATE -DMMDEPLOY_API_EXPORTS=1)
    mmdeploy_load_static(MMDeployExtern MMDeployStaticModules)
    mmdeploy_load_dynamic(MMDeployExtern MMDeployDynamicModules)
    target_link_libraries(MMDeployExtern PRIVATE MMDeployLibs)
endif ()
