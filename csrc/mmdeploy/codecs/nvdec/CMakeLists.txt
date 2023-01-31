# Copyright (c) OpenMMLab. All rights reserved.

if (NOT "cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
    return()
endif ()

project(mmdeploy_nvdec_decoder)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

set(SRCS
    video_decoder.cpp
    video_demuxer.cpp
    frame_queue.cpp)

mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")

# copy video codec sdk header to ${CUDA_INCLUDE_DIRS}
target_include_directories(${PROJECT_NAME}
    PRIVATE
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
)

# copy the nvcuvid stub lib to ${CUDA_LIBRARIES}
get_filename_component(_CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
target_link_directories(${PROJECT_NAME} PRIVATE ${_CUDA_LIB_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${CUDA_LIBRARIES} nvcuvid Threads::Threads ${OpenCV_LIBS} nppc nppicc)
