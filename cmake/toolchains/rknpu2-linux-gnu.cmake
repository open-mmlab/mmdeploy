set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR rockchip)

if(DEFINED ENV{RKNN_TOOL_CHAIN})
    file(TO_CMAKE_PATH $ENV{RKNN_TOOL_CHAIN} RKNN_TOOL_CHAIN)
else()
    message(FATAL_ERROR "RKNN_TOOL_CHAIN env must be defined")
endif()

set(CMAKE_C_COMPILER ${RKNN_TOOL_CHAIN}/bin/aarch64-rockchip-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${RKNN_TOOL_CHAIN}/bin/aarch64-rockchip-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_C_FLAGS "-Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "-Wl,--allow-shlib-undefined")

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")
