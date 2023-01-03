if (NOT ("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES))
    return()
endif ()

project(mmdeploy_operation_cuda CUDA CXX)

find_package(pplcv REQUIRED)

set(SRCS resize.cpp
        cvtcolor.cpp
        pad.cpp
        to_float.cpp
        cast.cu
        hwc2chw.cpp
        transpose.cu
        normalize.cpp
        normalize.cu
        crop.cpp
        crop.cu
        flip.cpp
        warp_affine.cpp)

mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")

target_link_libraries(${PROJECT_NAME} PRIVATE
        mmdeploy_operation
        ${PPLCV_LIBRARIES})
target_include_directories(${PROJECT_NAME}
        PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include ${PPLCV_INCLUDE_DIRS})
