# Usage

**step 1.** Run the demo in the console.

Use **ImageClassification** as example.

First, you should set your model path and image path to `${MODEL_PATH}` and `${IMAGE_PATH}`. And then follow the bash codes.

```bash
export TASK=ImageClassification
export LD_LIBRARY_PATH=${PWD}/build/lib:${LD_LIBRARY_PATH}
cd demo/java
java -cp ../../csrc/mmdeploy/apis/java ${TASK}.java ${MODEL_PATH} cpu ${IMAGE_PATH}
```
