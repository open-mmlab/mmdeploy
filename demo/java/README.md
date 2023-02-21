# Usage

**step 1.** Compile Java API classes.

```
cd ../../csrc/mmdeploy/apis/java
javac mmdeploy/*.java
cd ../..
```

**step 2.** Build the demo java project by Ant.

Use **ImageClassification** as example.

First, you should set your mmdeploy path, opencv path, model path and image path to `${MMDeploy_DIR}`, `${OPENCV_DIR}` `${MODEL_PATH}` and `${IMAGE_PATH}`. And then follow the bash codes.

```bash
export TASK=ImageClassification
export ARGS=${TASK}.java\ cpu\ ${MODEL_PATH}\ ${IMAGE_PATH}
ant -DtaskName=${TASK} -DjarDir=${OPENCV_DIR}/build/bin -DlibDir=${OPENCV_DIR}/build/lib:${MMDeploy_DIR}/build/lib -Dcommand=${ARGS}
```

As for **PoseTrack**, the ARGS format should be changed to:

```bash
export ARGS=${TASK}.java\ cpu\ ${DET_MODEL_PATH}\  ${POSE_MODEL_PATH}\ ${VIDEO_PATH}
```
