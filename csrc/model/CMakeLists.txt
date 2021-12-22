# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(model)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)

set(MODEL_NAMES "directory_model")
if (${MMDEPLOY_ZIP_MODEL})
    set(MODEL_NAMES ${MODEL_NAMES} "zip_model")
endif ()

foreach (MODEL_NAME ${MODEL_NAMES})
    set(TARGET_MODEL_NAME mmdeploy_${MODEL_NAME})
    build_target(${TARGET_MODEL_NAME} ${MODEL_NAME}_impl.cpp)
     target_link_libraries(${TARGET_MODEL_NAME}
            PRIVATE mmdeploy::core
            PUBLIC stdc++fs)
    if (${MODEL_NAME} STREQUAL "zip_model")
        find_package(libzip QUIET)
        if (libzip_FOUND)
            target_link_libraries(${TARGET_MODEL_NAME} PUBLIC libzip::zip)
        else()
            target_link_libraries(${TARGET_MODEL_NAME} PUBLIC zip)
        endif()
    endif ()
    add_library(mmdeploy::${MODEL_NAME} ALIAS ${TARGET_MODEL_NAME})
    export_module(${TARGET_MODEL_NAME})
endforeach ()
