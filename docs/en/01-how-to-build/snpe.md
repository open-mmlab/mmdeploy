# Build for SNPE

It is quit simple to support snpe backend: Client/Servcer mode.

this mode

1. Can split `model convert` and `inference` environments;

- Inference irrelevant matters are done on host
- We can get the real running results of gpu/npu instead of cpu simulation values

2. Can cover cost-sensitive device, armv7/risc-v/mips chips meet product requirements, but often have limited support for Python;

3. Can simplify mmdeploy installation steps. If you only want to convert snpe model and test, you don't need to compile the .whl package.

## 1. Execute inference service

Download the prebuilt snpe inference service package, `adb push` it to the phone and execute.
Note that **the phone must have a qcom chip**.

```bash
$ wget https://media.githubusercontent.com/media/tpoisonooo/mmdeploy-onnx2ncnn-testdata/main/snpe-inference-server-1.59.tar
...
$ sudo apt install adb
$ adb push snpe-inference-server-1.59.tar  /data/local/tmp/

# decompress and execute
$ adb shell
venus:/ $ cd /data/local/tmp
130|venus:/data/local/tmp $ tar xvf snpe-inference-server-1.59.tar
...
130|venus:/data/local/tmp $ source export1.59.sh
130|venus:/data/local/tmp $ ./inference_server
...
  Server listening on [::]:50052
```

At this point the inference service should print all the ipv6 and ipv4 addresses of the device and listen on the port.

tips:

- If `adb devices` cannot find the device, may be:
  - Some cheap cables can only charge and cannot transmit data
  - or the "developer mode" of the phone is not turned on
- If you need to compile the binary by self, please refer to [NDK Cross Compiling snpe Inference Service](../appendix/cross_build_snpe_service.md)

## 2. Build mmdeploy

1. Environment

| Matters | Version            | Remarks                |
| ------- | ------------------ | ---------------------- |
| host OS | ubuntu18.04 x86_64 | snpe specified version |
| Python  | **3.6.0**          | snpe specified version |

2. Installation

If you only need to do model convert and test,  just `pip install` it.

```bash
$ cd /path/to/mmdeploy
$ python3 -m pip install -e .
```

## 3. Test the model

Take Resnet-18 as an example. First refer to [documentation to install mmcls](https://github.com/open-mmlab/mmclassification)  and  use `tools/deploy.py` to convert the model.

```bash
$ export MODEL_CONFIG=/path/to/mmclassification/configs/resnet/resnet18_8xb16_cifar10.py
$ export MODEL_PATH=https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth

# Convert the model
$ cd /path/to/mmdeploy
$ python3 tools/deploy.py  configs/mmcls/classification_snpe_dynamic.py $MODEL_CONFIG  $MODEL_PATH   /path/to/test.png   --work-dir resnet18   --device cpu  --uri 10.1.82.63\:50051

# Test
$ python3 tools/test.py configs/mmcls/classification_snpe_static.py   $MODEL_CONFIG    --model reset18/end2end.dlc   --metrics accuracy precision f1_score recall  --uri 10.1.82.63\:50051
```

Note that `--uri` is required to specify the ip and port of the snpe inference service, ipv4 and ipv6 addresses can be used.

## 4. Build SDK with Android SDK

If you also need to compile mmdeploy SDK with Android NDK, please continue reading.

### 1) Download NDK and OpenCV package and setup environment

```bash
# Download android OCV
$ export OPENCV_VERSION=4.5.4
$ wget https://github.com/opencv/opencv/releases/download/${OPENCV_VERSION}/opencv-${OPENCV_VERSION}-android-sdk.zip
$ unzip opencv-${OPENCV_VERSION}-android-sdk.zip

$ export ANDROID_OCV_ROOT=`realpath opencv-${OPENCV_VERSION}-android-sdk`

# Download ndk r17c
$ wget https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip
$ unzip android-ndk-r17c-linux-x86_64.zip

$ export ANDROID_NDK_ROOT=`realpath android-ndk-r17c`
```

### 2) Compile mmdeploy SDK

```bash
$ cd /path/to/mmdeploy
$ mkdir build && cd build
$ cmake .. \
  -DMMDEPLOY_BUILD_SDK=ON   -DMMDEPLOY_CODEBASES=all \
  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake \
  -DMMDEPLOY_CODEBASES=all  -DMMDEPLOY_TARGET_BACKENDS=snpe \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26  \
  -DANDROID_STL=c++_static  \
  -DOpenCV_DIR=${ANDROID_OCV_ROOT}/sdk/native/jni/abi-arm64-v8a \
  -DMMDEPLOY_SHARED_LIBS=ON

  $ make && make install
```

| Options                       | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| DMMDEPLOY_CODEBASES=all       | Compile all algorithms' post-process                         |
| CMAKE_TOOLCHAIN_FILE          | Load NDK parameters, mainly used to select compiler          |
| MMDEPLOY_TARGET_BACKENDS=snpe | Inference backend                                            |
| ANDROID_STL=c++\_static       | In case of NDK environment can not find suitable c++ library |
| MMDEPLOY_SHARED_LIBS=ON       | snpe does not provide static library                         |

### 3) Compile demo

```bash
$ cd /path/to/install/example
$ mkdir build && cd build

$ cmake .. \
  -DMMDEPLOY_BUILD_SDK=ON   -DMMDEPLOY_CODEBASES=all \
  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake \
  -DMMDEPLOY_CODEBASES=all  -DMMDEPLOY_TARGET_BACKENDS=snpe \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30  \
  -DANDROID_STL=c++_static  \
  -DOpenCV_DIR=${ANDROID_OCV_ROOT}/sdk/native/jni/abi-arm64-v8a \
  -DMMDEPLOY_SHARED_LIBS=ON \
  -DMMDeploy_DIR=${PWD}/../../lib/cmake/MMDeploy

$ make
$ tree -L 1
...
├── image_restorer
├── image_segmentation
├── object_detection
├── ocr
├── pose_detection
└── rotated_object_detection
```

Just `adb push` the binary file and .so to the device and execute.

### 4) Run the demo

First make sure that`--dump-info`is used during convert model, so that the `resnet18` directory has the files required by the SDK such as `pipeline.json`.

`adb push` the model directory, executable file and .so to the device.

```bash
$ cd /path/to/mmdeploy
$ adb push resnet18  /data/local/tmp

$ cd /path/to/install/
$ adb push lib /data/local/tmp

$ cd /path/to/install/example/build
$ adb push image_classification /data/local/tmp
```

Set up environment variable and execute the sample.

```bash
$ adb shell
venus:/ $ cd /data/local/tmp
venus:/data/local/tmp $ export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/data/local/tmp/lib

venus:/data/local/tmp $ ./image_classification cpu ./resnet18/  demo.JPEG
..
label: 2, score: 0.4355
```
