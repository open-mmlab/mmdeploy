# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

project(tritonmmdeploybackend LANGUAGES C CXX)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
# GPU support is disabled by default because recommended backend
# doesn't use GPUs.
#

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")


#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all
# repos that we depend on so that we can override the tags.
#
include(FetchContent)

FetchContent_Declare(
        repo-common
        GIT_REPOSITORY https://github.com/triton-inference-server/common.git
        GIT_TAG ${TRITON_COMMON_REPO_TAG}
        GIT_SHALLOW ON
)
FetchContent_Declare(
        repo-core
        GIT_REPOSITORY https://github.com/triton-inference-server/core.git
        GIT_TAG ${TRITON_CORE_REPO_TAG}
        GIT_SHALLOW ON
)
FetchContent_Declare(
        repo-backend
        GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
        GIT_TAG ${TRITON_BACKEND_REPO_TAG}
        GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

add_library(triton-mmdeploy-backend SHARED
        model_state.cpp
        instance_state.cpp
        convert.cpp
        mmdeploy.cpp)

target_include_directories(triton-mmdeploy-backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(
        triton-mmdeploy-backend PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -Wall -Wno-unused-parameter -Wno-type-limits -Werror>
        $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
)

target_link_libraries(
        triton-mmdeploy-backend
        PRIVATE
        triton-core-serverapi   # from repo-core
        triton-core-backendapi  # from repo-core
        triton-core-serverstub  # from repo-core
        triton-backend-utils    # from repo-backend
)

mmdeploy_load_static(triton-mmdeploy-backend MMDeployStaticModules)
target_link_libraries(triton-mmdeploy-backend PRIVATE MMDeployLibs)

mmdeploy_export(triton-mmdeploy-backend)


if (WIN32)
    set_target_properties(
            triton-mmdeploy-backend PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            OUTPUT_NAME triton_mmdeploy
    )
else ()
    set_target_properties(
            triton-mmdeploy-backend PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            OUTPUT_NAME triton_mmdeploy
    )
endif ()
