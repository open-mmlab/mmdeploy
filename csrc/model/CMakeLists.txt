# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(model)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)

set(TARGET_NAMES "directory_model")
if (${MMDEPLOY_ZIP_MODEL})
    set(TARGET_NAMES ${TARGET_NAMES} "zip_model")
endif ()

foreach (TARGET_NAME ${TARGET_NAMES})
    set_targets("mmdeploy_${TARGET_NAME}" MODEL_OBJ MODEL_STATIC MODEL_SHARED)
    build_object_target(${MODEL_OBJ} "${TARGET_NAME}_impl.cpp")
    target_link_libraries(${MODEL_OBJ}
            PRIVATE mmdeploy::core::static
            PUBLIC stdc++fs)
    if (${TARGET_NAME} STREQUAL "zip_model")
        find_package(libzip QUIET)
        if (libzip_FOUND)
            target_link_libraries(${MODEL_OBJ} PUBLIC libzip::zip)
        else()
            target_link_libraries(${MODEL_OBJ} PUBLIC zip)
        endif()
    endif ()

    build_static_target(${MODEL_STATIC} ${MODEL_OBJ} "PUBLIC")
    add_library(mmdeploy::${TARGET_NAME}::static ALIAS ${MODEL_STATIC})

    build_shared_target(${MODEL_SHARED} ${MODEL_OBJ} "PUBLIC")
    add_library(mmdeploy::${TARGET_NAME} ALIAS ${MODEL_SHARED})

    export_module(${MODEL_STATIC} ${MODEL_SHARED} ${MODEL_OBJ})
endforeach ()
