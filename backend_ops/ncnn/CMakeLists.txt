set(TARGET_NAME mmlab_ncnn_ops)
set(SHARED_TARGET ${TARGET_NAME})

# ncnn
set(ncnn_DIR ${NCNN_DIR}/build/install/lib/cmake/ncnn)
find_package(ncnn)

if(ncnn_FOUND)
  message(STATUS "ncnn library found!")
else()
  message(FATAL_ERROR "Could not locate ncnn")
endif()

if(NOT ANDROID AND NOT IOS)
  add_subdirectory(ops)
  add_subdirectory(onnx2ncnn)
  add_subdirectory(pyncnn_ext)
else()
  # In case of embedded platform, like android, or ios, we only build custom ncnn
  # ops, and leave the executable converter(onnx2ncnn, pyncnn_ext) built under
  # the host platforms
  add_subdirectory(ops)
endif()
