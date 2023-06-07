# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_model)

set(MODEL_NAMES "directory_model")
if (${MMDEPLOY_ZIP_MODEL})
    set(MODEL_NAMES ${MODEL_NAMES} "zip_model")

    include(ExternalProject)
    set(LIBZIP_BUILD_DIR ${CMAKE_BINARY_DIR}/libzip-build)
    set(LIBZIP_INSTALL_DIR ${CMAKE_BINARY_DIR}/libzip-install)
    set(LIBZIP_INCLUDE_DIR ${LIBZIP_INSTALL_DIR}/include)
    set(LIBZIP_LIBRARY_DIR ${LIBZIP_INSTALL_DIR}/lib)
    ExternalProject_Add(
        libzip-external
        GIT_REPOSITORY https://github.com/nih-at/libzip
        GIT_TAG v1.9.2
        CMAKE_ARGS
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
    )
    add_library(zip STATIC IMPORTED)
    set_target_properties(zip
        PROPERTIES IMPORTED_LOCATION ${LIBZIP_LIBRARY_DIR}/libzip.a)
    add_dependencies(zip libzip-external)
    set(ziplib zip z)
    install(FILES ${LIBZIP_LIBRARY_DIR}/libzip.a
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif ()

foreach (MODEL_NAME ${MODEL_NAMES})
    set(TARGET_MODEL_NAME mmdeploy_${MODEL_NAME})
    mmdeploy_add_module(${TARGET_MODEL_NAME} ${MODEL_NAME}_impl.cpp)
    if (${MODEL_NAME} STREQUAL "zip_model")
        target_link_libraries(${TARGET_MODEL_NAME} PRIVATE ${ziplib})
        target_link_directories(${TARGET_MODEL_NAME} INTERFACE
            $<INSTALL_INTERFACE:lib>)
        target_include_directories(${TARGET_MODEL_NAME} PRIVATE
            $<BUILD_INTERFACE:${LIBZIP_INCLUDE_DIR}>)
    endif ()
    add_library(mmdeploy::${MODEL_NAME} ALIAS ${TARGET_MODEL_NAME})
endforeach ()
