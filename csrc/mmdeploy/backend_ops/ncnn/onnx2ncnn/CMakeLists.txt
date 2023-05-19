# Copyright (c) OpenMMLab. All rights reserved.

project(onnx2ncnn)

find_package(Protobuf)

if (PROTOBUF_FOUND)
    if (${Protobuf_PROTOC_EXECUTABLE} STREQUAL "")
        message(FATAL_ERROR "protoc not found, try `-DProtobuf_PROTOC_EXECUTABLE=/path/to/protoc`")
    endif ()
    protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS
            ${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto)
    add_executable(mmdeploy_onnx2ncnn onnx2ncnn.cpp fuse_pass.cpp shape_inference.cpp ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})
    target_include_directories(mmdeploy_onnx2ncnn PRIVATE ${PROTOBUF_INCLUDE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(mmdeploy_onnx2ncnn PRIVATE ${PROTOBUF_LIBRARIES})
    if (MSVC)
        target_compile_options(mmdeploy_onnx2ncnn PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/Za>)
    endif()
    set(_NCNN_CONVERTER_DIR ${CMAKE_SOURCE_DIR}/mmdeploy/backend/ncnn)
    install(TARGETS mmdeploy_onnx2ncnn DESTINATION ${_NCNN_CONVERTER_DIR})
else ()
    message(
            FATAL_ERROR "Protobuf not found, onnx model convert tool won't be built")
endif ()
