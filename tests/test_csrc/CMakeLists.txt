# Copyright (c) OpenMMLab. All rights reserved.

project(tests)

set(TC_SRCS test_main.cpp)

set(NET_IPU ${CMAKE_CURRENT_SOURCE_DIR}/net/test_ipu_net.cpp)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/archive ARCHIVE_TC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core CORE_TC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/preprocess TRANSFORM_TC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/net NET_TC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/model MODEL_TC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/graph GRAPH_TC)

set(DEVICE_TC)
foreach (DEVICE IN LISTS MMDEPLOY_TARGET_DEVICES)
    list(APPEND DEVICE_TC
            ${CMAKE_CURRENT_SOURCE_DIR}/device/test_${DEVICE}_device.cpp)
endforeach ()

set(CAPI_TC)
if ("all" IN_LIST MMDEPLOY_CODEBASES)
    set(TASK_LIST
            "classifier;detector;segmentor;text_detector;text_recognizer;restorer;model"
            )
    set(CODEBASES "mmcls;mmdet;mmseg;mmedit;mmocr")
else ()
    set(TASK_LIST "model")
    set(CODEBASES "${MMDEPLOY_CODEBASES}")
    if ("mmcls" IN_LIST MMDEPLOY_CODEBASES)
        list(APPEND TASK_LIST "classifier")
    endif ()
    if ("mmdet" IN_LIST MMDEPLOY_CODEBASES)
        list(APPEND TASK_LIST "detector")
    endif ()
    if ("mmseg" IN_LIST MMDEPLOY_CODEBASES)
        list(APPEND TASK_LIST "segmentor")
    endif ()
    if ("mmedit" IN_LIST MMDEPLOY_CODEBASES)
        list(APPEND TASK_LIST "restorer")
    endif ()
    if ("mmocr" IN_LIST MMDEPLOY_CODEBASES)
        list(APPEND TASK_LIST "text_detector")
        list(APPEND TASK_LIST "text_recognizer")
    endif ()
endif ()
foreach (TASK ${TASK_LIST})
    list(APPEND CAPI_TC ${CMAKE_CURRENT_SOURCE_DIR}/capi/test_${TASK}.cpp)
endforeach ()

# generate the header file
configure_file(config/test_define.h.in
        ${CMAKE_CURRENT_SOURCE_DIR}/test_define.h)

set(NET_TESTS
        ${NET_IPU}
        ${TC_SRCS}
        # ${ARCHIVE_TC}
        # ${CORE_TC}
        # ${TRANSFORM_TC}
        # ${MODEL_TC}
        
        # ${DEVICE_TC}
        # ${CAPI_TC}
        )

set(TC_SRCS
        ${NET_TC}
        ${TC_SRCS}
        ${ARCHIVE_TC}
        ${CORE_TC}
        ${TRANSFORM_TC}
        ${MODEL_TC}
        
        ${DEVICE_TC}
        ${CAPI_TC}
        ${GRAPH_TC})

add_executable(mmdeploy_tests ${TC_SRCS})
target_include_directories(mmdeploy_tests
        PRIVATE ${CMAKE_SOURCE_DIR}/third_party/catch2)
target_include_directories(mmdeploy_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (NOT (MMDEPLOY_SHARED_LIBS OR MSVC))
    target_compile_options(mmdeploy_tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)
endif ()

mmdeploy_load_static(mmdeploy_tests MMDeployStaticModules)
mmdeploy_load_dynamic(mmdeploy_tests MMDeployDynamicModules)
target_link_libraries(mmdeploy_tests PRIVATE
    MMDeployLibs
    mmdeploy_transform
    mmdeploy_opencv_utils)

add_executable(mmdeploy_tests_ipu ${NET_TESTS})
target_include_directories(mmdeploy_tests_ipu
        PRIVATE ${CMAKE_SOURCE_DIR}/third_party/catch2)
target_include_directories(mmdeploy_tests_ipu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (NOT (MMDEPLOY_SHARED_LIBS OR MSVC))
    target_compile_options(mmdeploy_tests_ipu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)
endif ()

mmdeploy_load_static(mmdeploy_tests_ipu MMDeployStaticModules)
mmdeploy_load_dynamic(mmdeploy_tests_ipu MMDeployDynamicModules)
target_link_libraries(mmdeploy_tests_ipu PRIVATE
    MMDeployLibs
    mmdeploy_transform
    mmdeploy_opencv_utils 
    )
