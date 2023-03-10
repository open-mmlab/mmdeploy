# Copyright (c) OpenMMLab. All rights reserved.
project(mmdeploy_java)

if (NOT ANDROID)
        find_package(JNI REQUIRED)
else ()
        set(JNI_LIBRARIES)
endif()

mmdeploy_add_library(${PROJECT_NAME} SHARED EXCLUDE
        mmdeploy_Classifier.cpp
        mmdeploy_Detector.cpp
        mmdeploy_Segmentor.cpp
        mmdeploy_Restorer.cpp
        mmdeploy_PoseDetector.cpp
        mmdeploy_TextDetector.cpp
        mmdeploy_TextRecognizer.cpp
        mmdeploy_PoseTracker.cpp
        mmdeploy_Context.cpp
        mmdeploy_Device.cpp
        mmdeploy_Model.cpp
        mmdeploy_Profiler.cpp
        mmdeploy_Scheduler.cpp
        mmdeploy_RotatedDetector.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${JNI_INCLUDE_DIRS})

mmdeploy_load_static(${PROJECT_NAME} MMDeployStaticModules)
mmdeploy_load_dynamic(${PROJECT_NAME} MMDeployDynamicModules)

target_link_libraries(${PROJECT_NAME} PRIVATE
        ${JNI_LIBRARIES} MMDeployLibs)
install(TARGETS ${PROJECT_NAME}
            DESTINATION lib)
