cmake_minimum_required(VERSION 3.14)
project(RISCVServer C CXX)
include(./common.cmake)
include(../../cmake/MMDeploy.cmake)

# Proto file
get_filename_component(rv_proto "inference.proto" ABSOLUTE)
get_filename_component(rv_proto_path "${rv_proto}" PATH)

# Generated sources
set(rv_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/inference.pb.cc")
set(rv_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/inference.pb.h")
set(rv_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/inference.grpc.pb.cc")
set(rv_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/inference.grpc.pb.h")

add_custom_command(
      OUTPUT "${rv_proto_srcs}" "${rv_proto_hdrs}" "${rv_grpc_srcs}" "${rv_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${rv_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${rv_proto}"
      DEPENDS "${rv_proto}")

find_package(ncnn REQUIRED)
# find_package(pplnn REQUIRED)

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_executable(inference_server
    server_impl.cpp
    inference_server.cpp
    backend.cpp
    ${rv_proto_srcs}
    ${rv_grpc_srcs})

target_link_libraries(inference_server
    PRIVATE
    mmdeploy_ncnn_ops_obj
    mmdeploy::core
    ncnn
    ${PPLNN_LIBRARIES}
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF}
    -static)

set_target_properties(inference_server
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/riscv-server)

# generate pb file
# GRPC_BIN_PATH=/opt/grpc/build/install/bin
# ${GRPC_BIN_PATH}/protoc -I . --python_out client --grpc_out client inference.proto --plugin=protoc-gen-grpc=${GRPC_BIN_PATH}/grpc_python_plugin
