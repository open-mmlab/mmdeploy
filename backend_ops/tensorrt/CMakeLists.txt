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

# cub
if (NOT DEFINED CUB_ROOT_DIR)
  set(CUB_ROOT_DIR "${PROJECT_SOURCE_DIR}/third_party/cub")
endif()
INCLUDE_DIRECTORIES(${CUB_ROOT_DIR})

# add plugin source
set(PLUGIN_LISTS scatternd
                 nms
                 roi_align
                 batched_nms
                 instance_norm
                 multi_level_roi_align)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
        file(GLOB PLUGIN_OPS_SRCS ${PLUGIN_ITER}/*.cpp ${PLUGIN_ITER}/*.cu)
        file(GLOB PLUGIN_OPS_HEADS ${PLUGIN_ITER}/*.h ${PLUGIN_ITER}/*.hpp ${PLUGIN_ITER}/*.cuh)
        set(BACKEND_OPS_SRCS ${BACKEND_OPS_SRCS} ${PLUGIN_OPS_SRCS} ${PLUGIN_OPS_HEADS})
endforeach(PLUGIN_ITER)

list(APPEND BACKEND_OPS_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/common_impl/trt_cuda_helper.cu")

set(INFER_PLUGIN_LIB ${TENSORRT_LIBRARY})

cuda_add_library(${SHARED_TARGET} MODULE ${BACKEND_OPS_SRCS})
target_link_libraries(${SHARED_TARGET} ${INFER_PLUGIN_LIB})
target_include_directories(${SHARED_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
