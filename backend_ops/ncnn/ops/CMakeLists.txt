# add plugin source
set(PLUGIN_LISTS custom_reshape
                 gather
                 shape)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    file(GLOB PLUGIN_OPS_SRCS ${PLUGIN_ITER}/*.cpp)
    file(GLOB PLUGIN_OPS_HEADS ${PLUGIN_ITER}/*.h)
    set(BACKEND_OPS_SRCS ${BACKEND_OPS_SRCS} ${PLUGIN_OPS_SRCS} ${PLUGIN_OPS_HEADS})
endforeach(PLUGIN_ITER)

set(BACKEND_OPS_SRCS ${BACKEND_OPS_SRCS} ncnn_ops_register.cpp)

add_library(${SHARED_TARGET} SHARED ${BACKEND_OPS_SRCS})
target_link_libraries(${SHARED_TARGET} ncnn)
