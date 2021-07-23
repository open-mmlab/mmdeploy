set(TARGET_NAME mmlab_onnxruntime_ops)
set(SHARED_TARGET ${TARGET_NAME})

# onnxruntime
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)


# add plugin source
set(PLUGIN_LISTS grid_sample
                 roi_align)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    file(GLOB PLUGIN_OPS_SRCS ${PLUGIN_ITER}/*.cpp ${PLUGIN_ITER}/*.cu)
    file(GLOB PLUGIN_OPS_HEADS ${PLUGIN_ITER}/*.h ${PLUGIN_ITER}/*.hpp ${PLUGIN_ITER}/*.cuh)
    set(BACKEND_OPS_SRCS ${BACKEND_OPS_SRCS} ${PLUGIN_OPS_SRCS} ${PLUGIN_OPS_HEADS})
endforeach(PLUGIN_ITER)

file(GLOB COMMON_SRCS common/*.cpp common/*.cu)
set(BACKEND_OPS_SRCS ${BACKEND_OPS_SRCS} ${COMMON_SRCS})

set(INFER_PLUGIN_LIB onnxruntime)
list(APPEND BACKEND_OPS_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_register.cpp")

add_library(${SHARED_TARGET} SHARED ${BACKEND_OPS_SRCS})
target_link_libraries(${SHARED_TARGET} ${INFER_PLUGIN_LIB})
target_include_directories(${SHARED_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
