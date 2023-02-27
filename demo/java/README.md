# Usage

**step 1.** Install Apache Ant.

Please check the Apache Ant existence using `ant -h` in the command line.

If there is no Apache Ant installed, please follow the command below.

```
sudo apt-get update
sudo apt-get install ant
```

Set environment variable

```
export ANT_HOME=/usr/share/ant
export PATH=${ANT_HOME}/bin:${PATH}
```

**step 2.** Build OpenCV jar package (PoseTracker only).

PoseTracker demo needs OpenCV Java, if you are interested in PoseTracker demo, you need to build OpenCV jar package first.

Using OpenCV-4.7.0 as example:

```
export OPENCV_VERSION=4.7.0
export JAVA_AWT_INCLUDE_PATH=${JAVA_HOME}
export JAVA_AWT_LIBRARY=${JAVA_HOME}
export JAVA_INCLUDE_PATH=${JAVA_HOME}/include
export JAVA_INCLUDE_PATH2=${JAVA_HOME}/include/darwin
export JAVA_JVM_LIBRARY=${JAVA_HOME}

wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
unzip ${OPENCV_VERSION}.zip && rm -rf ${OPENCV_VERSION}.zip && cd opencv-${OPENCV_VERSION}
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_JAVA=ON ..
make -j8 && make install
```

**step 3.** Build MMDeploy Java API.

Please check the existence of `*.class` in `${MMDEPLOY_DIR}/csrc/mmdeploy/apis/java/mmdeploy`.

If there is no existence of `*.class`, please follow this [tutorial](../../csrc/mmdeploy/apis/java/README.md) to build Java class.

**step 4.** Build demo.

Use **ImageClassification** as example.

First, you should set your mmdeploy path, opencv path, model path and image path to `${MMDEPLOY_DIR}`, `${OPENCV_DIR}`, `${MODEL_PATH}` and `${IMAGE_PATH}`. And then follow the bash codes.

```bash
cd demo/java
export TASK=ImageClassification
export ARGS=cpu\ ${MODEL_PATH}\ ${IMAGE_PATH}
ant -DtaskName=${TASK} -DjarDir=${OPENCV_DIR}/build/bin -DlibDir=${OPENCV_DIR}/build/lib:${MMDEPLOY_DIR}/build/lib -Dcommand=${ARGS}
```

As for **PoseTracker**, you should execute:

```bash
cd demo/java
export TASK=PoseTracker
export ARGS=cpu\ ${DET_MODEL_PATH}\  ${POSE_MODEL_PATH}\ ${VIDEO_PATH}
ant -DtaskName=${TASK} -DjarDir=${OPENCV_DIR}/build/bin -DlibDir=${OPENCV_DIR}/build/lib:${MMDEPLOY_DIR}/build/lib -Dcommand="${ARGS}"
```
