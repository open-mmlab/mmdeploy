# Usage

**step 1.** Compile Utils with Java APIs.

```
cd demo/java
javac --class-path ../../csrc/mmdeploy/apis/java/ Utils.java
cd ../..
```

**step 2.** Run the demo in the console.

Use **ImageClassification** as example.

First, you should set your model path and image path to `${MODEL_PATH}` and `${IMAGE_PATH}`. And then follow the bash codes.

```bash
export TASK=ImageClassification
export LD_LIBRARY_PATH=${PWD}/build/lib:${LD_LIBRARY_PATH}
cd demo/java
java -cp ../../csrc/mmdeploy/apis/java:./ ${TASK}.java cpu ${MODEL_PATH} ${IMAGE_PATH}
```
