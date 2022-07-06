# Coverage settings - compiler options

find_program(LLVM_COV_PATH "llvm-cov")
find_program(LCOV_PATH "lcov")
find_program(GENHTML_PATH "genhtml")

if("${CMAKE_C_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang")
    message("Building with llvm-cov gcov and lcov")

    # Warning/Error messages
    if(NOT LLVM_COV_PATH)
        message(FATAL_ERROR "llvm-cov not found! Aborting.")
    endif()
elseif(CMAKE_COMPILER_IS_GNUCXX)
    message("Building with GNU gcov and lcov")
else()
    message(FATAL_ERROR "Code coverage requires Clang or GCC. Aborting.")
endif()

# Warning/Error messages
if(NOT (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    message(WARNING "Code coverage results with an optimized (non-Debug) build may be misleading")
endif()
if(NOT LCOV_PATH)
    message(FATAL_ERROR "lcov not found! Aborting...")
endif()
if(NOT GENHTML_PATH)
    message(FATAL_ERROR "genhtml not found! Aborting...")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")

add_custom_target(lcov-gen
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/coverage"
    COMMAND ${LCOV_PATH} -d "${CMAKE_BINARY_DIR}" -c -o "${CMAKE_BINARY_DIR}/coverage/full.info"
    VERBATIM COMMAND ${LCOV_PATH} -r "${CMAKE_BINARY_DIR}/coverage/full.info" "*/usr/include/*" "*/googletest/*" "*massert*" "*/tests/*" -o "${CMAKE_BINARY_DIR}/coverage/out.info"
    )

add_custom_target(coverage-report
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/html"
    COMMAND ${GENHTML_PATH} -o "${CMAKE_BINARY_DIR}/html" "${CMAKE_BINARY_DIR}/coverage/out.info"
    COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan "Coverage report HTML at ${CMAKE_BINARY_DIR}/html/index.html"
    DEPENDS lcov-gen
    )
