# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_model)

set(MODEL_NAMES "directory_model")
if (${MMDEPLOY_ZIP_MODEL})
    set(MODEL_NAMES ${MODEL_NAMES} "zip_model")

    if (MSVC)
        set(zlib zlibstatic)
        set(LIB_PREFIX "")
        set(LIB_SUFFIX ".lib")
    else()
        set(zlib z)
        set(LIB_PREFIX "lib")
        set(LIB_SUFFIX ".a")
    endif()
    set(ziplib zip)

    set(zlib_name ${LIB_PREFIX}${zlib}${LIB_SUFFIX})
    set(ziplib_name ${LIB_PREFIX}${ziplib}${LIB_SUFFIX})

    include(ExternalProject)
    set(ZLIB_BUILD_DIR ${CMAKE_BINARY_DIR}/zlib-build)
    set(ZLIB_INSTALL_DIR ${CMAKE_BINARY_DIR}/zlib-install)
    set(ZLIB_INCLUDE_DIR ${ZLIB_INSTALL_DIR}/include)
    set(ZLIB_LIBRARY_DIR ${ZLIB_INSTALL_DIR}/lib)
    ExternalProject_Add(
        zlib-external
        GIT_REPOSITORY https://github.com/madler/zlib
        GIT_TAG v1.2.13
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${ZLIB_INSTALL_DIR}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        UPDATE_DISCONNECTED 1
    )
    add_custom_target(static_zlib ALL
        COMMAND ${CMAKE_COMMAND} -E remove ${ZLIB_LIBRARY_DIR}/zlib.lib
        COMMAND ${CMAKE_COMMAND} -E remove ${ZLIB_LIBRARY_DIR}/libz.so
        COMMAND ${CMAKE_COMMAND} -E remove ${ZLIB_LIBRARY_DIR}/libz.dylib
    )
    add_dependencies(static_zlib zlib-external)

    add_library(${zlib} STATIC IMPORTED)
    set_target_properties(${zlib}
        PROPERTIES IMPORTED_LOCATION ${ZLIB_LIBRARY_DIR}/${zlib_name})
    add_dependencies(${zlib} zlib-external)

    set(LIBZIP_BUILD_DIR ${CMAKE_BINARY_DIR}/libzip-build)
    set(LIBZIP_INSTALL_DIR ${CMAKE_BINARY_DIR}/libzip-install)
    set(LIBZIP_INCLUDE_DIR ${LIBZIP_INSTALL_DIR}/include)
    set(LIBZIP_LIBRARY_DIR ${LIBZIP_INSTALL_DIR}/lib)
    ExternalProject_Add(
        libzip-external
        GIT_REPOSITORY https://github.com/nih-at/libzip
        GIT_TAG v1.9.2
        CMAKE_ARGS
        -DZLIB_ROOT=${ZLIB_INSTALL_DIR}
        -DCMAKE_INSTALL_PREFIX=${LIBZIP_INSTALL_DIR}
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DBUILD_SHARED_LIBS=OFF
        -DLIBZIP_DO_INSTALL=ON
        -DBUILD_TOOLS=OFF
        -DBUILD_DOC=OFF
        -DBUILD_REGRESS=OFF
        -DBUILD_EXAMPLES=OFF
        -DENABLE_OPENSSL=OFF
        -DENABLE_COMMONCRYPTO=OFF
        -DENABLE_GNUTLS=OFF
        -DENABLE_MBEDTLS=OFF
        -DENABLE_WINDOWS_CRYPTO=OFF
        -DENABLE_BZIP2=OFF
        -DENABLE_LZMA=OFF
        -DENABLE_ZSTD=OFF
        PREFIX libzip
        BINARY_DIR ${LIBZIP_BUILD_DIR}
        # INSTALL_COMMAND ""
        UPDATE_DISCONNECTED 1
    )
    add_library(${ziplib} STATIC IMPORTED)
    set_target_properties(${ziplib}
        PROPERTIES IMPORTED_LOCATION ${LIBZIP_LIBRARY_DIR}/${ziplib_name})
    add_dependencies(libzip-external static_zlib)
    add_dependencies(${ziplib} libzip-external)
    set(ziplibs ${ziplib} ${zlib})
    if (NOT MMDEPLOY_BUILD_SDK_MONOLITHIC)
        install(FILES ${LIBZIP_LIBRARY_DIR}/${ziplib_name}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
        install(FILES ${ZLIB_LIBRARY_DIR}/${zlib_name}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
    endif()
endif ()

foreach (MODEL_NAME ${MODEL_NAMES})
    set(TARGET_MODEL_NAME mmdeploy_${MODEL_NAME})
    mmdeploy_add_module(${TARGET_MODEL_NAME} ${MODEL_NAME}_impl.cpp)
    if (${MODEL_NAME} STREQUAL "zip_model")
        target_link_libraries(${TARGET_MODEL_NAME} PRIVATE ${ziplibs})
        target_link_directories(${TARGET_MODEL_NAME} INTERFACE
            $<INSTALL_INTERFACE:lib>)
        target_include_directories(${TARGET_MODEL_NAME} PRIVATE
            $<BUILD_INTERFACE:${LIBZIP_INCLUDE_DIR}>)
    endif ()
    add_library(mmdeploy::${MODEL_NAME} ALIAS ${TARGET_MODEL_NAME})
endforeach ()
