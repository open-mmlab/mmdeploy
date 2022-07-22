cmake_minimum_required(VERSION 3.8)
project(Elena LANGUAGES CXX)

if (${CMAKE_VERSION} VERSION_GREATER "3.12.4")
    cmake_policy(SET CMP0074 NEW)
endif()

# Utility Functions
include(cmake/utils/Util.cmake)
include(cmake/utils/FindCUDA.cmake)

# Options
elena_option(ELENA_EXAMPLE "Build MMDeploy examples" ON)
elena_option(ELENA_COVERAGE "Build lcov coverage" OFF)
elena_option(ELENA_USE_CUDA "Build with CUDA" ON)

# 3rdparty
elena_option(ELENA_JSONCPP_PATH "Path to jsoncpp" 3rdparty/jsoncpp)

# Across Platform
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++ -O2 -fPIC ${CMAKE_CXX_FLAGS}")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++ -O2 -fPIC ${CMAKE_CXX_FLAGS}")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -fPIC ${CMAKE_CXX_FLAGS}")
else ()
    message(FATAL_ERROR "${CMAKE_CXX_COMPILER_ID} is unsupported.")
endif()

# Within Parrots or not
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
    set(CMAKE_CXX_FLAGS "-w ${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "-Wall -fno-rtti ${CMAKE_CXX_FLAGS}")
endif()


# build type option
if (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE ReleaseWithDebInfo)
endif()

# elena 3rdparty jsoncpp
add_subdirectory(./3rdparty/jsoncpp)
include_directories(./3rdparty/jsoncpp/include)
set(ELENA_JSON_LIB jsoncpp_lib)

# elena src (Note that all the src files should be listed explicitly.)
file(GLOB ELENA_SRC
    "src/*.cpp"
    "src/CodeGen/*.cpp"
    "src/IR/*.cpp"
    "src/Pass/Common/*.cpp"
    "src/Pass/Common/Simplify/*.cpp"
    "src/Pass/Hardware/*.cpp"
    "src/Schedule/*.cpp"
)

# elena include
include_directories(include)

# Python Libraries
if (${CMAKE_VERSION} VERSION_GREATER "3.12.4")
    find_package(Python3 REQUIRED Development Interpreter)
else ()
    find_package(PythonLibs 3.6 REQUIRED)
    set(Python3_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS})
    set(Python3_LIBRARIES ${PYTHON_LIBRARIES})
endif()

include_directories(${Python3_INCLUDE_DIRS})

# elena library.
if (hasParent)
    set(ELENA_LIB_STATIC elena_static)
    add_library(${ELENA_LIB_STATIC} STATIC ${ELENA_SRC} )
    target_link_libraries(${ELENA_LIB_STATIC} ${ELENA_JSON_LIB} ${Python3_LIBRARIES})
else()
    set(ELENA_LIB elena)
    add_library(${ELENA_LIB} SHARED ${ELENA_SRC})
    target_link_libraries(${ELENA_LIB} ${ELENA_JSON_LIB} ${Python3_LIBRARIES})

    if (ELENA_EXAMPLE)
        add_subdirectory(examples/MMDeploy)
    endif()

    if(ELENA_COVERAGE)
        include(cmake/coverage.cmake)
    endif()

    # cmake module
    include(cmake/modules/CUDA.cmake)

endif()
