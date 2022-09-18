# Build for SNPE

It is quite simple to support snpe backend: Client/Server mode.

this mode

1. Can split `model convert` and `inference` environments;

- Inference irrelevant matters are done on host
- We can get the real running results of gpu/npu instead of cpu simulation values

2. Can cover cost-sensitive device, armv7/risc-v/mips chips meet product requirements, but often have limited support for Python;

3. Can simplify mmdeploy installation steps. If you only want to convert snpe model and test, you don't need to compile the .whl package.

## 1. Run inference server

Download the prebuilt snpe inference server package, `adb push` it to the phone and execute.
Note that **the phone must have a qcom chip**.

```bash
$ wget https://media.githubusercontent.com/media/tpoisonooo/mmdeploy_snpe_testdata/main/snpe-inference-server-1.59.tar.gz
...
$ sudo apt install adb
$ adb push snpe-inference-server-1.59.tar.gz  /data/local/tmp/

# decompress and execute
$ adb shell
venus:/ $ cd /data/local/tmp
130|venus:/data/local/tmp $ tar xvf snpe-inference-server-1.59.tar.gz
...
130|venus:/data/local/tmp $ source export1.59.sh
130|venus:/data/local/tmp $ ./inference_server
...
  Server listening on [::]:60000
```

At this point the inference service should print all the ipv6 and ipv4 addresses of the device and listen on the port.

tips:

- If `adb devices` cannot find the device, may be:
  - Some cheap cables can only charge and cannot transmit data
  - or the "developer mode" of the phone is not turned on
- If you need to compile the binary by self, please refer to [NDK Cross Compiling snpe Inference Service](../appendix/cross_build_snpe_service.md)
- If a `segmentation fault` occurs when listening on a port, it may be because:
  - The port number is already occupied, use another port

## 2. Build mmdeploy

### 1) Environment

| Matters | Version            | Remarks                |
| ------- | ------------------ | ---------------------- |
| host OS | ubuntu18.04 x86_64 | snpe specified version |
| Python  | **3.6.0**          | snpe specified version |

### 2) Installation

Download [snpe-1.59 from the official website](https://developer.qualcomm.com/qfile/69652/snpe-1.59.0.zip)

```bash
$ unzip snpe-1.59.0.zip
$ export SNPE_ROOT=${PWD}/snpe-1.59.0.3230
$ cd /path/to/mmdeploy
$ export PYTHONPATH=${PWD}/service/snpe/client:${SNPE_ROOT}/lib/python:${PYTHONPATH}
$ export LD_LIBRARY_PATH=${SNPE_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}
$ export PATH=${SNPE_ROOT}/bin/x86_64-linux-clang:${PATH}
$ python3 -m pip install -e .
```

## 3. Test the model

Take Resnet-18 as an example. First refer to [documentation to install mmcls](https://github.com/open-mmlab/mmclassification)  and  use `tools/deploy.py` to convert the model.

```bash
$ export MODEL_CONFIG=/path/to/mmclassification/configs/resnet/resnet18_8xb16_cifar10.py
$ export MODEL_PATH=https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth

# Convert the model
$ cd /path/to/mmdeploy
$ python3 tools/deploy.py  configs/mmcls/classification_snpe_static.py $MODEL_CONFIG  $MODEL_PATH   /path/to/test.png   --work-dir resnet18   --device cpu  --uri 10.0.0.1\:60000  --dump-info

# Test
$ python3 tools/test.py configs/mmcls/classification_snpe_static.py   $MODEL_CONFIG    --model reset18/end2end.dlc   --metrics accuracy precision f1_score recall  --uri 10.0.0.1\:60000
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

# Download ndk r23b
$ wget https://dl.google.com/android/repository/android-ndk-r23b-linux.zip
$ unzip android-ndk-r23b-linux.zip

$ export ANDROID_NDK_ROOT=`realpath android-ndk-r23b`
```

### 2) Compile mmdeploy SDK and demo

```bash
$ cd /path/to/mmdeploy
$ mkdir build && cd build
$ cmake .. \
  -DMMDEPLOY_BUILD_SDK=ON \
  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake \
  -DMMDEPLOY_TARGET_BACKENDS=snpe \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30  \
  -DANDROID_STL=c++_static  \
  -DOpenCV_DIR=${ANDROID_OCV_ROOT}/sdk/native/jni/abi-arm64-v8a \
  -DMMDEPLOY_BUILD_EXAMPLES=ON

  $ make && make install
  $ tree ./bin
./bin
├── image_classification
├── image_restorer
├── image_segmentation
├── mmdeploy_onnx2ncnn
├── object_detection
├── ocr
├── pose_detection
└── rotated_object_detection
```

| Options                       | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| CMAKE_TOOLCHAIN_FILE          | Load NDK parameters, mainly used to select compiler          |
| MMDEPLOY_TARGET_BACKENDS=snpe | Inference backend                                            |
| ANDROID_STL=c++\_static       | In case of NDK environment can not find suitable c++ library |
| MMDEPLOY_SHARED_LIBS=ON       | snpe does not provide static library                         |

[Here](../01-how-to-build/cmake_option.md) is all cmake build option description.

### 3) Run the demo

First make sure that`--dump-info`is used during convert model, so that the `resnet18` directory has the files required by the SDK such as `pipeline.json`.

`adb push` the model directory, executable file and .so to the device.

```bash
$ cd /path/to/mmdeploy
$ adb push resnet18  /data/local/tmp
$ adb push tests/data/tiger.jpeg /data/local/tmp/resnet18/

$ cd /path/to/install/
$ adb push lib /data/local/tmp
$ adb push bin/image_classification /data/local/tmp/resnet18/
```

Set up environment variable and execute the sample.

```bash
$ adb push /path/to/mmcls/demo/demo.JPEG /data/local/tmp
$ adb shell
venus:/ $ cd /data/local/tmp/resnet18
venus:/data/local/tmp/resnet18 $ export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/data/local/tmp/lib

venus:/data/local/tmp/resnet18 $ ./image_classification cpu ./  tiger.jpeg
..
label: 3, score: 0.3214
```
