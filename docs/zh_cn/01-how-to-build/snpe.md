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
$ wget https://media.githubusercontent.com/media/tpoisonooo/mmdeploy-onnx2ncnn-testdata/main/snpe-inference-server-1.59.tar
...
$ sudo apt install adb
$ adb push snpe-inference-server-1.59.tar  /data/local/tmp/

# 解压运行
$ adb shell
venus:/ $ cd /data/local/tmp
130|venus:/data/local/tmp $ tar xvf snpe-inference-server-1.59.tar
...
130|venus:/data/local/tmp $ source export1.59.sh
130|venus:/data/local/tmp $ ./inference_server
...
listening 
...
```
此时推理服务应打印设备所有 ipv6 和 ipv4 地址，并监听 `50051` 端口。

tips:

* 如果 `adb devices` 找不到设备，可能因为：
  * 有些廉价线只能充电、不能传输数据
  * 或者没有打开手机的“开发者模式”
* 如果需要自己编译，可参照 [NDK 交叉编译 snpe 推理服务](../06-appendix/cross-build-ndk-gRPC.md) 

## 二、安装 mmdeploy

1. 环境要求

| 事项    | 版本               | 备注              |
| ------- | ------------------ | ----------------- |
| host OS | ubuntu18.04 x86_64 | snpe 指定版本 |
| Python  | **3.6.0**              | snpe 指定版本 |

2. 安装

如果只需要做模型转换和精度测试，`pip` 安装即可。
```bash
$ cd /path/to/mmdeploy
$ python3 -m pip install -e .
```

## 三、测试模型

以 Resnet-18 为例。先参照[文档安装 mmcls](https://github.com/open-mmlab/mmclassification)，然后使用 `tools/deploy.py` 转换模型。

```bash
$ export MODEL_CONFIG=/path/to/mmclassification/configs/resnet/resnet18_8xb16_cifar10.py
$ export MODEL_PATH=https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth

# 模型转换
$ cd /path/to/mmdeploy
$ python3 tools/deploy.py  configs/mmcls/classification_snpe_dynamic.py $MODEL_CONFIG  $MODEL_PATH   /path/to/test.png   --work-dir resnet18   --device cpu  --uri 10.1.82.63\:50051

# 精度测试
$ python3 tools/test.py configs/mmcls/classification_snpe_dynamic.py   $MODEL_CONFIG    --model reset18/end2end.dlc   --metrics accuracy precision f1_score recall  --uri 10.1.82.63\:50051
```

注意需要 `--uri` 指明 snpe 推理服务的 ip 和端口号，可以使用 ipv4 和 ipv6 地址。

## 四、Android NDK 编译 SDK

如果你还需要用 Android NDK 编译 mmdeploy SDK，请继续阅读本章节。

1. 下载 OCV 和 NDK 包，设置环境变量

```bash
# 下载 android OCV
$ export OPENCV_VERSION=4.5.4
$ wget https://github.com/opencv/opencv/releases/download/${OPENCV_VERSION}/opencv-${OPENCV_VERSION}-android-sdk.zip
$ unzip opencv-${OPENCV_VERSION}-android-sdk.zip

$ export ANDROID_OCV_ROOT=`realpath opencv-${OPENCV_VERSION}-android-sdk`

# 下载 ndk r17c
$ wget https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip
$ unzip android-ndk-r17c-linux-x86_64.zip

$ export ANDROID_NDK_ROOT=`realpath android-ndk-r17c`
```

2. 编译 mmdeploy snpe SDK

```bash
$ cd /path/to/mmdeploy
$ mkdir build && cd build
$ cmake 
```

