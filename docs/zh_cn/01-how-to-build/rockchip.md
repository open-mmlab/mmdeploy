# 瑞芯微 NPU 部署

- [模型转换](#模型转换)
  - [安装环境](#安装环境)
  - [分裂模型转换](#分类模型转换)
  - [检测模型转换](#检测模型转换)
- [模型推理](#模型推理)
  - [Host 交叉编译](#Host-交叉编译)
  - [Device 执行推理](#Device-执行推理)

______________________________________________________________________

MMDeploy 支持把模型部署到瑞芯微设备上。已支持的芯片：RV1126、RK3588。

完整的部署过程包含两个步骤：

1. 模型转换

   - 在主机上，将 PyTorch 模型转换为 RKNN 模型

2. 模型推理

   - 在主机上， 使用交叉编译工具得到设备所需的 SDK 和 bin
   - 把转好的模型和编好的 SDK、bin，传到设备，进行推理

## 模型转换

### 安装环境

1. 请参考[快速入门](../get_started.md)，创建 conda 虚拟环境，并安装 PyTorch、mmcv-full

2. 安装 RKNN Toolkit

   如下表所示，瑞芯微提供了 2 套 RKNN Toolkit，对应于不同的芯片型号

   <table>
    <thead>
      <tr>
        <th>Device</th>
        <th>RKNN-Toolkit</th>
        <th>Installation Guide</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>RK1808 / RK1806 / RV1109 / RV1126</td>
        <td><code>git clone https://github.com/rockchip-linux/rknn-toolkit</code></td>
        <td><a href="https://github.com/rockchip-linux/rknn-toolkit2/tree/master/doc">安装指南</a></td>
      </tr>
      <tr>
        <td>RK3566 / RK3568 / RK3588 / RV1103 / RV1106</td>
        <td><code>git clone https://github.com/rockchip-linux/rknn-toolkit2</code></td>
        <td><a href="https://github.com/rockchip-linux/rknn-toolkit/tree/master/doc">安装指南</a></td>
      </tr>
    </tbody>
    </table>

   2.1 通过 `git clone` 下载和设备匹配的 RKNN Toolkit

   2.2 参考表中的安装指南，安装 RKNN python 安装包。建议在安装时，使用选项 `--no-deps`，以避免依赖包的冲突。以 rknn-toolkit2 为例:

   ```
   pip install packages/rknn_toolkit2-1.2.0_f7bb160f-cp36-cp36m-linux_x86_64.whl --no-deps
   ```

   2.3 先安装onnx==1.8.0,跟着 [instructions](../01-how-to-build/build_from_source.md)，源码安装 MMDeploy。 需要注意的是， MMDeploy 和 RKNN 依赖的安装包间有冲突的内容. 这里提供建议在 python 3.6 环境中使用的安装包版本:

   ```
   protobuf==3.19.4
   onnx==1.8.0
   onnxruntime==1.8.0
   torch==1.8.0
   torchvision==0.9.0
   ```

### 分类模型转换

以 mmclassification 中的 resnet50 为例，模型转换命令如下：

```shell
# 安装 mmclassification
pip install mmcls
git clone https://github.com/open-mmlab/mmclassification

# 执行转换命令
cd /the/path/of/mmdeploy
python tools/deploy.py \
    configs/mmcls/classification_rknn_static.py \
    /the/path/of/mmclassification/configs/resnet/resnet50_8xb32_in1k.py \
    https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth \
    /the/path/of/mmclassification/demo/demo.JPEG \
    --work-dir mmdeploy_models/mmcls/resnet50 \
    --device cpu \
    --dump-info
```

```{note}
若转换过程中，遇到 NoModuleFoundError 的问题，使用 pip install 对应的包
```

### 检测模型转换

- YOLOV3 & YOLOX

将下面的模型拆分配置写入到 [detection_rknn_static.py](https://github.com/open-mmlab/mmdeploy/blob/master/configs/mmdet/detection/detection_rknn_static.py)

```python
# yolov3, yolox for rknn-toolkit and rknn-toolkit2
partition_config = dict(
  type='rknn',  # the partition policy name
  apply_marks=True,  # should always be set to True
  partition_cfg=[
      dict(
          save_file='model.onnx',  # name to save the partitioned onnx
          start=['detector_forward:input'],  # [mark_name:input, ...]
          end=['yolo_head:input'],  # [mark_name:output, ...]
          output_names=[f'pred_maps.{i}' for i in range(3)]) # output names
  ])
```

执行命令：

```shell
# 安装 mmdet
pip install mmdet
git clone https://github.com/open-mmlab/mmdetection

# 执行转换命令
python tools/deploy.py \
    configs/mmcls/detection_rknn_static.py \

```

- RetinaNet & SSD & FSAF with rknn-toolkit2

将下面的模型拆分配置写入到 [detection_rknn_static.py](https://github.com/open-mmlab/mmdeploy/blob/master/configs/mmdet/detection/detection_rknn_static.py)。使用 rknn-toolkit 的用户则不用。

```python
# retinanet, ssd
partition_config = dict(
    type='rknn',  # the partition policy name
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file='model.onnx',
            start='detector_forward:input',
            end=['BaseDenseHead:output'],
            output_names=[f'BaseDenseHead.cls.{i}' for i in range(5)] +
            [f'BaseDenseHead.loc.{i}' for i in range(5)])
    ])
```

### 部署 config 说明

部署 config，你可以根据需要修改 `backend_config` 字段. 一个 mmclassification 的 `backend_config`例子如下:

```python
backend_config = dict(
    type='rknn',
    common_config=dict(
        mean_values=None,
        std_values=None,
        target_platform='rk3588',
        optimization_level=3),
    quantization_config=dict(do_quantization=False, dataset=None),
    input_size_list=[[3, 224, 224]])

```

`common_config` 的内容服务于 `rknn.config()`. `quantization_config` 的内容服务于 `rknn.build()`。

### 问题说明

- SDK 只支持 int8 的 rknn 模型，这需要在转换模型时设置 `do_quantization=True`。

## 模型推理

### Host 交叉编译

若 host 是 Ubuntu 18.04 及以上版本，推荐脚本编译：

```shell
bash tools/scripts/ubuntu_cross_build_rknn.sh <model>
```

命令中的参数 model 表示瑞芯微芯片的型号，目前支持 rv1126，rk3588。

以下是对脚本中编译过程的说明。

如下表所示，瑞芯微提供了 2 套 RKNN API 工具包，对应于不同的芯片型号。而每套 RKNN API 工具包又分别对应不同的 gcc 交叉编译工具。

| Device                                     | RKNN API                                           |
| ------------------------------------------ | -------------------------------------------------- |
| RK1808 / RK1806 / RV1109 / RV1126          | [rknpu](https://github.com/rockchip-linux/rknpu)   |
| RK3566 / RK3568 / RK3588 / RV1103 / RV1106 | [rknpu2](https://github.com/rockchip-linux/rknpu2) |

以支持的 rv1126 和 rk3588 为例，mmdeploy 在 ubuntu18.04 上的交叉编译过程如下：

- **rv11126**

1. 下载 RKNN API 包

```shell
git clone https://github.com/rockchip-linux/rknpu
export RKNPU_DIR=$(pwd)/rknpu
```

2. 准备 gcc 交叉编译工具

```shell
sudo apt-get update
sudo apt-get install gcc-arm-linux-gnueabihf
sudo apt-get install g++-arm-linux-gnueabihf
```

3. 源码安装 OpenCV

```shell
git clone https://github.com/opencv/opencv --depth=1 --branch=4.6.0 --recursive
cd opencv
mkdir -p build_arm_gnueabi && cd build_arm_gnueabi
cmake .. -DCMAKE_INSTALL_PREFIX=install \
    -DCMAKE_TOOLCHAIN_FILE=../platforms/linux/arm-gnueabi.toolchain.cmake \
    -DBUILD_PERF_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
make -j $(nproc) && make install
export OpenCV_ARM_INSTALL_DIR=$(pwd)/install
```

4. 编译 mmdeploy SDK

```shell
cd /path/to/mmdeploy
mkdir -p build && cd build
cmake .. \
-DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/arm-linux-gnueabihf.cmake \
-DMMDEPLOY_BUILD_SDK=ON \
-DMMDEPLOY_BUILD_SDK_CXX_API=ON \
-DMMDEPLOY_BUILD_EXAMPLES=ON \
-DMMDEPLOY_TARGET_BACKENDS="rknn" \
-DRKNPU_DEVICE_DIR=${RKNPU_DIR}/rknn/rknn_api/librknn_api \
-DOpenCV_DIR=${OpenCV_ARM_INSTALL_DIR}/lib/cmake/opencv4
make -j$(nproc) && make install
```

- **rk3588**

1. 下载 RKNN API 包

```shell
git clone https://github.com/rockchip-linux/rknpu2
export RKNPU2_DEVICE_DIR=$(pwd)/rknpu2/runtime/RK3588
```

2. 准备 gcc 交叉编译工具

```shell
git clone https://github.com/Caesar-github/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu
export RKNN_TOOL_CHAIN=$(pwd)/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu
export LD_LIBRARY_PATH=$RKNN_TOOL_CHAIN/lib64:$LD_LIBRARY_PATH
```

3. 下载 opencv 预编译包

```shell
git clone https://github.com/opencv/opencv --depth=1 --branch=4.6.0 --recursive
cd opencv
mkdir -p build_aarch64 && cd build_aarch64
cmake .. -DCMAKE_INSTALL_PREFIX=install
    -DCMAKE_TOOLCHAIN_FILE=../platforms/linux/aarch64-gnu.toolchain.cmake \
    -DBUILD_PERF_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
make -j $(nproc) && make install
export OpenCV_AARCH64_INSTALL_DIR=$(pwd)/install
```

4. 编译 mmdeploy SDK

```shell
cd /path/to/mmdeploy
mkdir -p build && cd build
export LD_LIBRARY_PATH=$RKNN_TOOL_CHAIN/lib64:$LD_LIBRARY_PATH
cmake \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/rknpu2-linux-gnu.cmake \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DMMDEPLOY_BUILD_SDK_CXX_API=ON \
    -DMMDEPLOY_TARGET_BACKENDS="rknn" \
    -DMMDEPLOY_BUILD_EXAMPLES=ON \
    -DOpenCV_DIR=${OpenCV_AARCH64_INSTALL_DIR}/lib/cmake/opencv4
make -j $(nproc) && make install
```

### Device 执行推理

首先，确保`--dump-info`在转模型的时候调用了, 这样工作目录下包含 SDK 需要的配置文件 `pipeline.json`。

使用 `adb push` 将转好的模型、编好的 SDK 和 bin 文件推到设备上。

```bash
cd {/the/path/to/mmdeploy}
adb push mmdeploy_models/mmcls/resnet50  /root/resnet50
adb push {/the/path/of/mmclassification}/demo/demo.JPEG /root/demo.JPEG
adb push build/install /root/mmdeploy_sdk
```

通过 adb shell，打开设备终端，设置环境变量，运行例子。

```bash
adb shell
cd /root/mmdeploy_sdk
export LD_LIBRARY_PATH=$(pwd)/lib:${LD_LIBRARY_PATH}
./bin/image_classification cpu ../resnet50  ../demo.JPEG
```

结果显示：

```shell
label: 65, score: 0.95
```
