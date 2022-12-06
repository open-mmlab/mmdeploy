# ubuntu 交叉编译 aarch64

mmdeploy 选 ncnn 作为 aarch64 嵌入式 linux 设备的推理后端。 完整的部署分为两部分：

Host

- 模型转换
- 交叉编译嵌入式设备所需 SDK 和 bin

Device

- 运行编译结果

## 1. Host 模型转换

参照文档安装 [mmdeploy](../01-how-to-build/) 和 [mmcls](https://github.com/open-mmlab/mmclassification)，转换 resnet18 对应模型包

```bash
export MODEL_CONFIG=/path/to/mmclassification/configs/resnet/resnet18_8xb32_in1k.py
export MODEL_PATH=https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth

# 模型转换
cd /path/to/mmdeploy
python tools/deploy.py \
  configs/mmcls/classification_ncnn_static.py \
  $MODEL_CONFIG \
  $MODEL_PATH \
  tests/data/tiger.jpeg \
  --work-dir resnet18 \
  --device cpu \
  --dump-info
```

## 2. Host 交叉编译

建议直接用脚本编译

```bash
sh -x tools/scripts/ubuntu_cross_build_aarch64.sh
```

以下是脚本对应的手动过程

a) 安装 aarch64 交叉编译工具

```bash
sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

b) 交叉编译 opencv 安装到 tmp 目录

```bash
git clone https://github.com/opencv/opencv --depth=1 --branch=4.x --recursive
cd opencv/platforms/linux/
mkdir build && cd build
cmake ../../.. \
  -DCMAKE_INSTALL_PREFIX=/tmp/ocv-aarch64 \
  -DCMAKE_TOOLCHAIN_FILE=../aarch64-gnu.toolchain.cmake
make -j && make install
ls -alh /tmp/ocv-aarch64
..
```

c) 交叉编译 ncnn 安装到 tmp 目录

```bash
git clone https://github.com/tencent/ncnn --branch 20221128 --depth=1
mkdir build && cd build
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake \
  -DCMAKE_INSTALL_PREFIX=/tmp/ncnn-aarch64
make -j && make install
ls -alh /tmp/ncnn-aarch64
..
```

d) 交叉编译 mmdeploy，install/bin 目录是可执行文件

```bash
git submodule init
git submodule update
mkdir build && cd build
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/aarch64-linux-gnu.cmake \
  -DMMDEPLOY_TARGET_DEVICES="cpu" \
  -DMMDEPLOY_TARGET_BACKENDS="ncnn" \
  -Dncnn_DIR=/tmp/ncnn-aarch64/lib/cmake/ncnn \
  -DOpenCV_DIR=/tmp/ocv-aarch64/lib/cmake/opencv4
make install
ls -lah install/bin/*
..
```

## 3. Device 执行

确认转换模型用了 `--dump-info`，这样 `resnet18` 目录才有 `pipeline.json` 等 SDK 所需文件。

把 dump 好的模型目录(resnet18)、可执行文件(image_classification)、测试图片(tests/data/tiger.jpeg)、交叉编译的 OpenCV(/tmp/ocv-aarch64) 拷贝到设备中

```bash
./image_classification cpu ./resnet18  tiger.jpeg
..
label: 292, score: 0.9261
label: 282, score: 0.0726
label: 290, score: 0.0008
label: 281, score: 0.0002
label: 340, score: 0.0001
```
