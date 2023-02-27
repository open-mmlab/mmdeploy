if (NOT MMDEPLOY_BUILD_SDK_JAVA_API)
        return ()
endif ()

project(mmdeploy_java_package)

find_package(Java REQUIRED)
include(UseJava)

add_subdirectory(native)

add_jar(${PROJECT_NAME} SOURCES
        mmdeploy/DataType.java
        mmdeploy/Mat.java
        mmdeploy/InstanceMask.java
        mmdeploy/PixelFormat.java
        mmdeploy/PointF.java
        mmdeploy/Rect.java
        mmdeploy/Classifier.java
        mmdeploy/Detector.java
        mmdeploy/Segmentor.java
        mmdeploy/TextDetector.java
        mmdeploy/TextRecognizer.java
        mmdeploy/Restorer.java
        mmdeploy/PoseDetector.java
        mmdeploy/Context.java
        mmdeploy/Device.java
        mmdeploy/Model.java
        mmdeploy/Profiler.java
        mmdeploy/Scheduler.java
        mmdeploy/PoseTracker.java
        mmdeploy/RotatedDetector.java
        OUTPUT_NAME mmdeploy
        OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
