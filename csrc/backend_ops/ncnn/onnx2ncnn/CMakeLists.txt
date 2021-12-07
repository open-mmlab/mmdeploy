# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(onnx2ncnn)

find_package(Protobuf)

if (PROTOBUF_FOUND)
    protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS
            ${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto)
    add_executable(onnx2ncnn onnx2ncnn.cpp ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})
    target_include_directories(onnx2ncnn PRIVATE ${PROTOBUF_INCLUDE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(onnx2ncnn PRIVATE ${PROTOBUF_LIBRARIES})

else ()
    message(
            FATAL_ERROR "Protobuf not found, onnx model convert tool won't be built")
endif ()
