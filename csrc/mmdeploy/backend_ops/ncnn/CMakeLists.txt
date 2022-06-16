# Copyright (c) OpenMMLab. All rights reserved.

# ncnn
find_package(ncnn)

if (ncnn_FOUND)
    message(STATUS "ncnn library found!")
else ()
    message(FATAL_ERROR "Could not locate ncnn")
endif ()


if (NOT ANDROID AND NOT IOS)
    add_subdirectory(ops)
    add_subdirectory(onnx2ncnn)
    add_subdirectory(pyncnn_ext)
else ()
    # In case of embedded platform, like android, or ios, we only build custom ncnn
    # ops, and leave the executable converter(onnx2ncnn, pyncnn_ext) built under
    # the host platforms
    add_subdirectory(ops)
endif ()
