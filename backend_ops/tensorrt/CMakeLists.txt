set(TARGET_NAME mmlab_tensorrt_ops)
set(SHARED_TARGET ${TARGET_NAME})

# cuda
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES(/usr/local/cuda/include)
enable_language(CUDA)

# tensorrt
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
if (TENSORRT_INCLUDE_DIR)
    MESSAGE(STATUS " Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
else()
    MESSAGE(ERROR " Cannot found TensorRT headers")
endif()

find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_DIR} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PARSERS nvparsers
        HINTS ${TENSORRT_DIR} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
        HINTS  ${TENSORRT_DIR} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} 
                     ${TENSORRT_LIBRARY_PARSERS} 
                     ${TENSORRT_LIBRARY_INFER_PLUGIN} 
                     )
if (TENSORRT_LIBRARY_INFER AND TENSORRT_LIBRARY_PARSERS AND TENSORRT_LIBRARY_INFER_PLUGIN)
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
else()
MESSAGE(ERROR " Cannot found TensorRT libs")
endif()
find_package_handle_standard_args(
        TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
    message(ERROR " Cannot find TensorRT library.")
endif()
INCLUDE_DIRECTORIES(${TENSORRT_INCLUDE_DIR})

# include common
include_directories(common)
add_subdirectory(common_impl)

# add plugin source
set(PLUGIN_LISTS scatternd)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)

list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/trt_plugin.cpp")

set(INFER_PLUGIN_LIB ${TENSORRT_LIBRARY})

cuda_add_library(${SHARED_TARGET} SHARED ${BACKEND_OPS_SRCS})
target_link_libraries(${SHARED_TARGET} ${INFER_PLUGIN_LIB})
target_include_directories(${SHARED_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
