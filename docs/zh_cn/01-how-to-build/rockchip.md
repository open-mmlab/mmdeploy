# 支持 RKNN

本教程基于 Ubuntu-18.04 和 Rockchip `rk3588` NPU。对于不同的 NPU 设备，您需要使用不同的 rknn 包.
这是设备和安装包的关系表:

| Device               | Python Package                                                   | c/c++ SDK                                          |
| -------------------- | ---------------------------------------------------------------- | -------------------------------------------------- |
| RK1808/RK1806        | [rknn-toolkit](https://github.com/rockchip-linux/rknn-toolkit)   | [rknpu](https://github.com/rockchip-linux/rknpu)   |
| RV1109/RV1126        | [rknn-toolkit](https://github.com/rockchip-linux/rknn-toolkit)   | [rknpu](https://github.com/rockchip-linux/rknpu)   |
| RK3566/RK3568/RK3588 | [rknn-toolkit2](https://github.com/rockchip-linux/rknn-toolkit2) | [rknpu2](https://github.com/rockchip-linux/rknpu2) |
| RV1103/RV1106        | [rknn-toolkit2](https://github.com/rockchip-linux/rknn-toolkit2) | [rknpu2](https://github.com/rockchip-linux/rknpu2) |

## 安装

建议为项目创建一个虚拟环境。

1. 使用 git 获取 RKNN-Toolkit2 或者 RKNN-Toolkit。以 RKNN-Toolkit2 为例:

   ```
   git clone git@github.com:rockchip-linux/rknn-toolkit2.git
   ```

2. 通过 [rknn-toolkit2 文档](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/doc) 或者 [rknn-toolkit 文档](https://github.com/rockchip-linux/rknn-toolkit/tree/master/doc)安装 RKNN python 安装包。安装 rknn python 包时，最好在安装命令后添加`--no-deps`，以避免依赖包的冲突。以rknn-toolkit2为例:

   ```
   pip install packages/rknn_toolkit2-1.2.0_f7bb160f-cp36-cp36m-linux_x86_64.whl --no-deps
   ```

3. 先安装onnx==1.8.0,跟着 [instructions](../01-how-to-build/build_from_source.md)，源码安装 MMDeploy。 需要注意的是， MMDeploy 和 RKNN 依赖的安装包间有冲突的内容. 这里提供建议在 python 3.6 环境中使用的安装包版本:

   ```
   protobuf==3.19.4
   onnx==1.8.0
   onnxruntime==1.8.0
   torch==1.8.0
   torchvision==0.9.0
   ```

4. 使用 conda 安装 torch and torchvision，比如:

```
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

如要使用 [MMClassification](https://mmclassification.readthedocs.io/en/latest/getting_started.html)， 需要用户自己安装使用。

## 使用

例子:

```bash
python tools/deploy.py \
    configs/mmcls/classification_rknn_static.py \
    /mmclassification_dir/configs/resnet/resnet50_8xb32_in1k.py \
    https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth \
    /mmclassification_dir/demo/demo.JPEG \
    --work-dir ../resnet50 \
    --device cpu
```

## 部署 config

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

## 安装 SDK

### RKNPU2 编译 MMDeploy SDK

1. 获取 rknpu2:

   ```
   git clone git@github.com:rockchip-linux/rknpu2.git
   ```

2. 在 linux 系统, 下载 gcc 交叉编译器. `rknpu2` 的官方提供的下载链接无法使用了. 用户可以使用另一个 [链接](https://github.com/Caesar-github/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu). 下载并解压完编译器, 打开终端, 设置 `RKNN_TOOL_CHAIN` 和 `RKNPU2_DEVICE_DIR` 为 `export RKNN_TOOL_CHAIN=/path/to/gcc/usr;export RKNPU2_DEVICE_DIR=/path/to/rknpu2/runtime/RK3588`。

3. 上述准备工作完成后, 运行如下指令安装:

```shell
cd /path/to/mmdeploy
mkdir -p build && rm -rf build/CM* && cd build
export LD_LIBRARY_PATH=$RKNN_TOOL_CHAIN/lib64:$LD_LIBRARY_PATH
cmake \
    -DCMAKE_TOOLCHAIN_FILE=/path/to/mmdeploy/cmake/toolchains/rknpu2-linux-gnu.cmake \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DOpenCV_DIR=${RKNPU2_DEVICE_DIR}/../../examples/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV \
    -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
    -DMMDEPLOY_TARGET_DEVICES="cpu" \
    -DMMDEPLOY_TARGET_BACKENDS="rknn" \
    -DMMDEPLOY_CODEBASES=all \
    -DMMDEPLOY_BUILD_TEST=ON \
    -DMMDEPLOY_BUILD_EXAMPLES=ON \
    ..
make && make install
```

## 运行 SDK 的 demo

首先，确保`--dump-info`在转模型的时候调用了, 这样工作目录下包含 SDK 需要的配置文件 `pipeline.json`。

使用 `adb push` 将模型路径，执行文件和.so 文件传到板子上。

```bash
cd /path/to/mmdeploy
adb push resnet50  /data/local/tmp/resnet50
adb push /mmclassification_dir/demo/demo.JPEG /data/local/tmp/resnet50/demo.JPEG
cd build
adb push lib /data/local/tmp/lib
adb push bin/image_classification /data/local/tmp/image_classification
```

设置环境变量，运行例子。

```bash
adb shell
cd /data/local/tmp
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/data/local/tmp/lib
./image_classification cpu ./resnet50  ./resnet50/demo.JPEG
..
label: 65, score: 0.95
```

## 问题点

- 量化失败.

  经验来说, 如果 `do_quantization` 被设置为 `True`，RKNN 需要的输入没有被归一化过。请修改 `Normalize` 在 `model_cfg` 的设置，如将

  ```python
  img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
  ```

  改为

  ```python
  img_norm_cfg = dict(
    mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
  ```

  此外, deploy_cfg 的 `mean_values` 和 `std_values` 应该被设置为 `model_cfg` 中归一化的设置. 使 `mean_values=[[103.53, 116.28, 123.675]]`, `std_values=[[57.375, 57.12, 58.395]]`。

- MMDet 模型.

  YOLOV3 & YOLOX: 将下面的模型拆分配置写入到 [detection_rknn_static.py](https://github.com/open-mmlab/mmdeploy/blob/master/configs/mmdet/detection/detection_rknn_static.py):

  ```python
  # yolov3, yolox
  partition_config = dict(
      type='rknn',  # the partition policy name
      apply_marks=True,  # should always be set to True
      partition_cfg=[
          dict(
              save_file='model.onnx',  # name to save the partitioned onnx
              start=['detector_forward:input'],  # [mark_name:input, ...]
              end=['yolo_head:input'])  # [mark_name:output, ...]
      ])
  ```

  RetinaNet & SSD & FSAF with rknn-toolkit2, 将下面的模型拆分配置写入到 [detection_rknn_static.py](https://github.com/open-mmlab/mmdeploy/blob/master/configs/mmdet/detection/detection_rknn_static.py)。使用 rknn-toolkit 的用户则不用。

  ```python
  # retinanet, ssd
  partition_config = dict(
      type='rknn',  # the partition policy name
      apply_marks=True,
      partition_cfg=[
          dict(
              save_file='model.onnx',
              start='detector_forward:input',
              end=['BaseDenseHead:output'])
      ])
  ```

- SDK 只支持 int8 的 rknn 模型，这需要在转换模型时设置 `do_quantization=True`。
