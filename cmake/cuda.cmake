# Copyright (c) OpenMMLab. All rights reserved.
#message(STATUS "CMAKE_VERSION: ${CMAKE_VERSION}")

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
    # suppress 'CMAKE_CUDA_ARCHITECTURES' warning
    cmake_policy(SET CMP0104 OLD)
endif ()

# nvcc compiler settings
find_package(CUDA REQUIRED)
#message(STATUS "CUDA VERSION: ${CUDA_VERSION_STRING}")

if (MSVC)
    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=/wd4819,/wd4828")
    if (HAVE_CXX_FLAG_UTF_8)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=/utf-8")
    endif ()
else ()
    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
    # Explicitly set the cuda host compiler. Because the default host compiler #
    # selected by cmake maybe wrong.
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CUDA_NVCC_FLAGS
            "${CUDA_NVCC_FLAGS} -Xcompiler=-fPIC,-Wall,-fvisibility=hidden")
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=-fno-gnu-unique")
    endif ()
endif ()

enable_language(CUDA)

# set virtual compute architecture and real ones
set(_NVCC_FLAGS)
if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_53,code=sm_53")
    if (CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "8")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    endif ()
    if (CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "9")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
    endif ()
    if (CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "10")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_72,code=sm_72")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
    endif ()
    if (CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "11")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
        if (CUDA_VERSION_MINOR VERSION_GREATER_EQUAL "1")
            # cuda doesn't support `sm_86` until version 11.1
            set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
        endif ()
    endif ()
endif ()

set(CUDA_NVCC_FLAGS_DEBUG "-g -O0")
set(CUDA_NVCC_FLAGS_RELEASE "-O3")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
if (NOT MSVC)
    set(CMAKE_CUDA_STANDARD 14)
endif ()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_NVCC_FLAGS} ${_NVCC_FLAGS}")
