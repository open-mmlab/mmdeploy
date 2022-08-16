set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv)

if(DEFINED ENV{RISCV_ROOT_PATH})
    file(TO_CMAKE_PATH $ENV{RISCV_ROOT_PATH} RISCV_ROOT_PATH)
else()
    message(FATAL_ERROR "RISCV_ROOT_PATH env must be defined")
endif()

set(CMAKE_C_COMPILER ${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-g++)

set(CMAKE_SYSROOT "${RISCV_ROOT_PATH}/sysroot" CACHE PATH "riscv sysroot")
set(CMAKE_FIND_ROOT_PATH ${RISCV_ROOT_PATH}/riscv64-unknown-linux-gnu)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_C_FLAGS "-march=rv64gc -mabi=lp64d -mtune=c906")
set(CMAKE_CXX_FLAGS "-march=rv64gc -mabi=lp64d -mtune=c906")
