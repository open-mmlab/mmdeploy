set(TARGET_NAME mmlab_onnxruntime_ops)
set(SHARED_TARGET ${TARGET_NAME})

# onnxruntime
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# add plugin source
file(GLOB_RECURSE BACKEND_OPS_SRCS *.cpp)

set(INFER_PLUGIN_LIB onnxruntime)

add_library(${SHARED_TARGET} SHARED ${BACKEND_OPS_SRCS})
target_link_libraries(${SHARED_TARGET} ${INFER_PLUGIN_LIB})
target_include_directories(${SHARED_TARGET}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
