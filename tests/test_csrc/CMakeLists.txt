# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(tests)

include(${CMAKE_SOURCE_DIR}/cmake/opencv.cmake)

# find TC source files and related shared libraries and modules that are going
# to be linked
set(MMDEPLOY_LIBS)
set(MMDEPLOY_MODULES mmdeploy::directory_model mmdeploy::transform_module
                     mmdeploy::transform mmdeploy::net_module mmdeploy::graph)
set(TC_SRCS test_main.cpp)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/archive ARCHIVE_TC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core CORE_TC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/preprocess TRANSFORM_TC)

set(DEVICE_TC)
foreach(DEVICE IN LISTS MMDEPLOY_TARGET_DEVICES)
  list(APPEND DEVICE_TC
       ${CMAKE_CURRENT_SOURCE_DIR}/device/test_${DEVICE}_device.cpp)
  list(APPEND MMDEPLOY_MODULES mmdeploy::device::${DEVICE}
       mmdeploy::transform_impl::${DEVICE})
endforeach()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/net NET_TC)
foreach(BACKEND IN LISTS MMDEPLOY_TARGET_BACKENDS)
  list(APPEND MMDEPLOY_MODULES mmdeploy::${BACKEND}_net)
endforeach()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/model MODEL_TC)
if(MMDEPLOY_ZIP_MODEL)
  message(STATUS "MMDEPLOY_ZIP_MODEL: ${MMDEPLOY_ZIP_MODEL}")
  list(APPEND MMDEPLOY_MODULES mmdeploy::zip_model)
endif()

set(CAPI_TC)
if("all" IN_LIST MMDEPLOY_CODEBASES)
  set(TASK_LIST
      "classifier;detector;segmentor;text_detector;text_recognizer;restorer;model"
  )
  set(CODEBASES "mmcls;mmdet;mmseg;mmedit;mmocr")
else()
  set(TASK_LIST "model")
  set(CODEBASES "${MMDEPLOY_CODEBASES}")
  if("mmcls" IN_LIST MMDEPLOY_CODEBASES)
    list(APPEND TASK_LIST "classifier")
  endif()
  if("mmdet" IN_LIST MMDEPLOY_CODEBASES)
    list(APPEND TASK_LIST "detector")
  endif()
  if("mmseg" IN_LIST MMDEPLOY_CODEBASES)
    list(APPEND TASK_LIST "segmentor")
  endif()
  if("mmedit" IN_LIST MMDEPLOY_CODEBASES)
    list(APPEND TASK_LIST "restorer")
  endif()
  if("mmocr" IN_LIST MMDEPLOY_CODEBASES)
    list(APPEND TASK_LIST "text_detector")
    list(APPEND TASK_LIST "text_recognizer")
  endif()
endif()
foreach(TASK ${TASK_LIST})
  list(APPEND CAPI_TC ${CMAKE_CURRENT_SOURCE_DIR}/capi/test_${TASK}.cpp)
  list(APPEND MMDEPLOY_LIBS mmdeploy_${TASK})
endforeach()

# generate the header file
configure_file(config/test_define.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/test_define.h)

set(TC_SRCS
    ${TC_SRCS}
    ${ARCHIVE_TC}
    ${CORE_TC}
    ${DEVICE_TC}
    ${CAPI_TC}
    ${TRANSFORM_TC}
    ${MODEL_TC}
    ${NET_TC})
list(APPEND MMDEPLOY_LIBS mmdeploy::core)
foreach(CODEBASE IN LISTS CODEBASES)
  list(APPEND MMDEPLOY_MODULES mmdeploy::${CODEBASE})
endforeach()

add_executable(mmdeploy_tests ${TC_SRCS})
target_include_directories(mmdeploy_tests
                           PRIVATE ${CMAKE_SOURCE_DIR}/third_party/catch2)
target_include_directories(mmdeploy_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  mmdeploy_tests PRIVATE ${MMDEPLOY_LIBS} ${OpenCV_LIBS} -Wl,--no-as-needed
                         ${MMDEPLOY_MODULES} -Wl,--as-need)
