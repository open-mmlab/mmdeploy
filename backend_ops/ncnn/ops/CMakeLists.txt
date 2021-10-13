# add plugin source
file(GLOB_RECURSE BACKEND_OPS_SRCS *.cpp)

add_library(${SHARED_TARGET} SHARED ${BACKEND_OPS_SRCS})
target_link_libraries(${SHARED_TARGET} ncnn)
