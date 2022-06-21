# Installation

## From Source

### Requirements

- OpenJDK >= 10

### Installation
**Step 1.** Generate java class and jni headers.

You should Generate java `.class` files and jni headers before build them with sdk.

```bash
cd csrc/mmdeploy/apis/java
javac -h native/ mmdeploy/*.java
cd ../../../..
```

**Step 2.** Build sdk.

After generate java class and jni headers, you should build them with sdk. Please follow this [tutorial](../../../../docs/en/01-how-to-build/linux-x86_64.md)/[教程](../../../../docs/zh_cn/01-how-to-build/linux-x86_64.md) to build sdk. Remember to set the MMDEPLOY_BUILD_SDK_JAVA_API option to ON.

**Step 3.** Run the demo in the console.

Now you can run the demo in the console. Use **ImageClassification** as example.

First, you should set your model path and image path to `${MODEL_PATH}` and `${IMAGE_PATH}`. And then follow the bash codes.

```bash
export TASK=ImageClassification
export LD_LIBRARY_PATH=${PWD}/build/install/lib:${LD_LIBRARY_PATH}
cd demo/java
java -cp ../../csrc/mmdeploy/apis/java/mmdeploy ${TASK}.java cpu ${MODEL_PATH} ${IMAGE_PATH}
```
