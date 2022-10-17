# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/pybind/pybind11/blob/master/tests/CMakeLists.txt

if (MSVC)
    set(STD_FS_NO_LIB_NEEDED TRUE)
else ()
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
            "#include <filesystem>\nint main(int,char**argv){return std::filesystem::path(argv[0]).string().length();}")
    try_compile(HAS_INC_FS ${CMAKE_CURRENT_BINARY_DIR}
            SOURCES ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
            COMPILE_DEFINITIONS -std=c++17 -c)

    if (NOT HAS_INC_FS)
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
                "#include <experimental/filesystem>\nint main(int,char**argv){return std::experimental::filesystem::path(argv[0]).string().length();}")
    endif ()

    try_compile(
            STD_FS_NO_LIB_NEEDED ${CMAKE_CURRENT_BINARY_DIR}
            SOURCES ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
            COMPILE_DEFINITIONS -std=c++17)
    try_compile(
            STD_FS_NEEDS_STDCXXFS ${CMAKE_CURRENT_BINARY_DIR}
            SOURCES ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
            COMPILE_DEFINITIONS -std=c++17
            LINK_LIBRARIES stdc++fs)
    try_compile(
            STD_FS_NEEDS_CXXFS ${CMAKE_CURRENT_BINARY_DIR}
            SOURCES ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
            COMPILE_DEFINITIONS -std=c++17
            LINK_LIBRARIES c++fs)
endif ()

if (${STD_FS_NO_LIB_NEEDED})
    set(STD_FS_LIB "")
elseif (${STD_FS_NEEDS_STDCXXFS})
    set(STD_FS_LIB stdc++fs)
elseif (${STD_FS_NEEDS_CXXFS})
    set(STD_FS_LIB c++fs)
else ()
    message(WARNING "Unknown C++17 compiler - not passing -lstdc++fs")
    set(STD_FS_LIB "")
endif ()
