add_definitions(-std=c++11)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# build ONNXRUNTIME ops
if (BUILD_ONNXRUNTIME_OPS)
    message("Build ONNXRUNTIME custom ops.")
    add_subdirectory (onnxruntime)
endif()

# build TensorRT ops
if (BUILD_TENSORRT_OPS)
    message("Build TensorRT custom ops.")
    add_subdirectory (tensorrt)
endif()
