set(TARGET_NAME mmlab_tensorrt_ops)
set(SHARED_TARGET ${TARGET_NAME})
set(STATIC_TARGET ${TARGET_NAME}_static)

# to suppress 'CMAKE_CUDA_ARCHITECTURES' warning when cmake version is >= 18.0
cmake_policy(SET CMP0104 OLD)
# cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
enable_language(CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=/wd4819,/wd4828")
  if(HAVE_CXX_FLAG_UTF_8)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=/utf-8")
  endif()
else()
  set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
  # Explicitly set the cuda host compiler. Because the default host compiler #
  # selected by cmake maybe wrong.
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  set(CUDA_NVCC_FLAGS
      "${CUDA_NVCC_FLAGS} -Xcompiler=-fPIC,-Wall,-fvisibility=hidden")
endif()

# set virtual compute architecture and real ones
set(_NVCC_FLAGS)
set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")
if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "8")
  set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
  set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
endif()
if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "9")
  set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
endif()
if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "10")
  set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_72,code=sm_72")
  set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
endif()
if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "11")
  set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
  set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
endif()

set(CUDA_NVCC_FLAGS_DEBUG "-g -O0")
set(CUDA_NVCC_FLAGS_RELEASE "-O3")
set(CUDA_NVCC_FLAGS "-std=c++11 ${CUDA_NVCC_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_NVCC_FLAGS} ${_NVCC_FLAGS}")

# tensorrt
find_path(
  TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
if(TENSORRT_INCLUDE_DIR)
  message(STATUS " Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
else()
  message(ERROR " Cannot found TensorRT headers")
endif()

find_library(
  TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_DIR} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(
  TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS ${TENSORRT_DIR} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER}
                     ${TENSORRT_LIBRARY_INFER_PLUGIN})
if(TENSORRT_LIBRARY_INFER
  AND TENSORRT_LIBRARY_INFER_PLUGIN)
  message(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
else()
  message(FATAL_ERROR " Cannot found TensorRT libs")
endif()
find_package_handle_standard_args(TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR
                                  TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR " Cannot find TensorRT library.")
endif()
include_directories(${TENSORRT_INCLUDE_DIR})

# cudnn
include_directories(${CUDNN_DIR}/include)
link_directories(${CUDNN_DIR}/lib64)

# cub
add_definitions(-DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT)
if(NOT DEFINED CUB_ROOT_DIR)
  set(CUB_ROOT_DIR "${PROJECT_SOURCE_DIR}/third_party/cub")
endif()
include_directories(${CUB_ROOT_DIR})

# add plugin source
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
file(GLOB_RECURSE BACKEND_OPS_SRCS *.cpp *.cu)

set(INFER_PLUGIN_LIB ${TENSORRT_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} cudnn)

cuda_add_library(${SHARED_TARGET} MODULE ${BACKEND_OPS_SRCS})
target_link_libraries(${SHARED_TARGET} ${INFER_PLUGIN_LIB})
target_include_directories(${SHARED_TARGET}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)

cuda_add_library(${STATIC_TARGET} STATIC ${BACKEND_OPS_SRCS})
target_link_libraries(${STATIC_TARGET} ${INFER_PLUGIN_LIB})
target_include_directories(${STATIC_TARGET}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
