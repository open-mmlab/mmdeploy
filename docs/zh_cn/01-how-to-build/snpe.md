# 支持 SNPE

mmdeploy 集成 snpe 的方式简单且有效： Client/Server 模式。

这种模式

1. 能剥离`模型转换`和`推理`环境：

- 推理无关事项在算力更高的设备上完成；
- 对于推理计算，能拿到 gpu/npu 真实运行结果，而非 cpu 模拟数值。

2. 能覆盖成本敏感的设备。armv7/risc-v/mips 芯片满足产品需求，但往往对 Python 支持有限；

3. 能简化 mmdeploy 安装步骤。如果只想转 snpe 模型测试精度，不需要编译 .whl 包。

## 一、运行推理服务

下载预编译 snpe 推理服务包， `adb push` 到手机、执行。
注意**手机要有 qcom 芯片**。

```bash
$ wget https://media.githubusercontent.com/media/tpoisonooo/mmdeploy_snpe_testdata/main/snpe-inference-server-1.59.tar.gz
...
$ sudo apt install adb
$ adb push snpe-inference-server-1.59.tar.gz  /data/local/tmp/

# 解压运行
$ adb shell
venus:/ $ cd /data/local/tmp
130|venus:/data/local/tmp $ tar xvf snpe-inference-server-1.59.tar.gz
...
130|venus:/data/local/tmp $ source export1.59.sh
130|venus:/data/local/tmp $ ./inference_server  60000
...
  Server listening on [::]:60000
```

此时推理服务应打印设备所有 ipv6 和 ipv4 地址，并监听端口。

tips:

- 如果 `adb devices` 找不到设备，可能因为：
  - 有些廉价线只能充电、不能传输数据
  - 或者没有打开手机的“开发者模式”
- 如果需要自己编译，可参照 [NDK 交叉编译 snpe 推理服务](../appendix/cross_build_snpe_service.md)
- 如果监听端口时  `segmentation fault`，可能是因为：
  - 端口号已占用，换一个端口

## 二、安装 mmdeploy

1. 环境要求

| 事项    | 版本               | 备注          |
| ------- | ------------------ | ------------- |
| host OS | ubuntu18.04 x86_64 | snpe 指定版本 |
| Python  | **3.6.0**          | snpe 指定版本 |

2. 安装

[官网下载 snpe-1.59](https://developer.qualcomm.com/qfile/69652/snpe-1.59.0.zip)，解压设置环境变量

```bash
$ unzip snpe-1.59.0.zip
$ export SNPE_ROOT=${PWD}/snpe-1.59.0.3230
$ cd /path/to/mmdeploy
$ export PYTHONPATH=${PWD}/service/snpe/client:${SNPE_ROOT}/lib/python:${PYTHONPATH}
$ export LD_LIBRARY_PATH=${SNPE_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}
$ export PATH=${SNPE_ROOT}/bin/x86_64-linux-clang:${PATH}
$ python3 -m pip install -e .
```

tips:

- 如果网络不好，[这个 .tar.gz](https://github.com/tpoisonooo/mmdeploy_snpe_testdata/blob/main/snpe-1.59.tar.gz) 仅减小官方包体积，没有修改原始内容。

## 三、测试模型

以 Resnet-18 为例。先参照[文档安装 mmcls](https://github.com/open-mmlab/mmclassification)，然后使用 `tools/deploy.py` 转换模型。

```bash
$ export MODEL_CONFIG=/path/to/mmclassification/configs/resnet/resnet18_8xb16_cifar10.py
$ export MODEL_PATH=https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth

# 模型转换
$ cd /path/to/mmdeploy
$ python3 tools/deploy.py  configs/mmcls/classification_snpe_static.py $MODEL_CONFIG  $MODEL_PATH   /path/to/test.png   --work-dir resnet18   --device cpu  --uri 192.168.1.1\:60000  --dump-info

# 精度测试
$ python3 tools/test.py configs/mmcls/classification_snpe_static.py   $MODEL_CONFIG    --model reset18/end2end.dlc   --metrics accuracy precision f1_score recall  --uri 192.168.1.1\:60000
```

注意需要 `--uri` 指明 snpe 推理服务的 ip 和端口号，可以使用 ipv4 和 ipv6 地址。

## 四、Android NDK 编译 SDK

如果你还需要用 Android NDK 编译 mmdeploy SDK，请继续阅读本章节。

### 1. 下载 OCV、NDK，设置环境变量

```bash
# 下载 android OCV
$ export OPENCV_VERSION=4.5.4
$ wget https://github.com/opencv/opencv/releases/download/${OPENCV_VERSION}/opencv-${OPENCV_VERSION}-android-sdk.zip
$ unzip opencv-${OPENCV_VERSION}-android-sdk.zip

$ export ANDROID_OCV_ROOT=`realpath opencv-${OPENCV_VERSION}-android-sdk`

# 下载 ndk r23b
$ wget https://dl.google.com/android/repository/android-ndk-r23b-linux.zip
$ unzip android-ndk-r23b-linux.zip

$ export ANDROID_NDK_ROOT=`realpath android-ndk-r23b`
```

### 2. 编译 mmdeploy SDK 和 demo

```bash
$ cd /path/to/mmdeploy
$ mkdir build && cd build
$ cmake .. \
  -DMMDEPLOY_BUILD_SDK=ON \
  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake \
  -DMMDEPLOY_TARGET_BACKENDS=snpe \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30  \
  -DANDROID_STL=c++_static \
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

选项说明

| 选项                          | 说明                                  |
| ----------------------------- | ------------------------------------- |
| CMAKE_TOOLCHAIN_FILE          | 加载 NDK 参数，主要用于选择编译器版本 |
| MMDEPLOY_TARGET_BACKENDS=snpe | 使用 snpe 推理                        |
| ANDROID_STL=c++\_static       | 避免 NDK 环境找不到合适的 c++ lib     |
| MMDEPLOY_SHARED_LIBS=ON       | 官方 snpe 没有提供静态库              |

[这里](../01-how-to-build/cmake_option.md)是完整的编译选项说明

## 3. 运行 demo

先确认测试模型用了 `--dump-info`，这样 `resnet18` 目录才有 `pipeline.json` 等 SDK 所需文件。

把 dump 好的模型目录、可执行文件和 lib 都 `adb push` 到设备里

```bash
$ cd /path/to/mmdeploy
$ adb push resnet18  /data/local/tmp
$ adb push tests/data/tiger.jpeg /data/local/tmp/resnet18/

$ cd /path/to/install/
$ adb push lib /data/local/tmp
$ adb push bin/image_classification /data/local/tmp/resnet18/
```

设置环境变量，执行样例

```bash
$ adb push /path/to/mmcls/demo/demo.JPEG /data/local/tmp
$ adb shell
venus:/ $ cd /data/local/tmp/resnet18
venus:/data/local/tmp/resnet18 $ export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/data/local/tmp/lib

venus:/data/local/tmp/resnet18 $ ./image_classification cpu ./  tiger.jpeg
..
label: 3, score: 0.3214
```
