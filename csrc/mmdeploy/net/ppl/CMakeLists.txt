# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_pplnn_net)

find_package(pplnn REQUIRED)

mmdeploy_add_module(${PROJECT_NAME} ppl_net.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${PPLNN_INCLUDE_DIRS}>)
if (PPLNN_USE_X86 AND ("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES))
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DPPL_NN_HAS_X86=1)
endif ()
if (PPLNN_USE_CUDA AND ("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES))
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DPPL_NN_HAS_CUDA=1)
    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/include)
    target_link_directories(${PROJECT_NAME} PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    target_link_libraries(${PROJECT_NAME} PRIVATE nvrtc)
endif ()
target_link_libraries(${PROJECT_NAME}
        PRIVATE ${PPLNN_LIBRARIES})
if (PPLNN_USE_RISCV AND ("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES))
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DPPL_NN_HAS_RISCV=1)
endif ()
add_library(mmdeploy::pplnn_net ALIAS ${PROJECT_NAME})
