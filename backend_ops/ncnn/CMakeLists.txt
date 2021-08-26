set(TARGET_NAME mmlab_ncnn_ops)
set(SHARED_TARGET ${TARGET_NAME})

# ncnn
set(ncnn_DIR ${NCNN_DIR}/build/install/lib/cmake/ncnn)
find_package(ncnn)

if (ncnn_FOUND)
    message(STATUS "ncnn library found!")
else ()
    message(FATAL_ERROR "Could not locate ncnn")
endif()

add_subdirectory (ops)
add_subdirectory (onnx2ncnn)
add_subdirectory (pyncnn_ext)
