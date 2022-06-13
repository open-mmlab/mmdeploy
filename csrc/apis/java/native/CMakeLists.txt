
project(mmdeploy_java)

find_package(JNI)

mmdeploy_add_library(${PROJECT_NAME} SHARED EXCLUDE
        mmdeploy_Classifier.cpp mmdeploy_Detector.cpp mmdeploy_Segmentor.cpp mmdeploy_Restorer.cpp mmdeploy_PoseDetector.cpp mmdeploy_RotatedDetector.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${JNI_INCLUDE_DIRS})

mmdeploy_load_static(${PROJECT_NAME} MMDeployStaticModules)
mmdeploy_load_dynamic(${PROJECT_NAME} MMDeployDynamicModules)
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${JNI_LIBRARIES} MMDeployLibs)
