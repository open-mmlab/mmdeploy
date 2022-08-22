# 支持 RISC-V

## 一、安装 mmdeploy

这里需要使用 mmdeploy_onnx2ncnn 进行模型转换，故需要安装 ncnn 推理引擎，可参考[BUILD 文档](./linux-x86_64.md) 进行安装，

## 二、测试模型

以 Resnet-18 为例。先参照[文档](https://github.com/open-mmlab/mmclassification)安装 mmcls，然后使用 `tools/deploy.py` 转换模型。

```bash
$ export MODEL_CONFIG=/path/to/mmclassification/configs/resnet/resnet18_8xb16_cifar10.py
$ export MODEL_PATH=https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth

export PYTHONPATH=${PWD}/service/riscv/client:${NCNN_ROOT}/build/python/ncnn:${PYTHONPATH}
export PATH=${pwd}/build/bin:${PATH}

# 模型转换
$ cd /path/to/mmdeploy
$ python3 tools/deploy.py configs/mmcls/classification_ncnn_static.py $MODEL_CONFIG  $MODEL_PATH   /path/to/test.png --work-dir resnet18 --device cpu --dump-info

# 精度测试
$ python3 tools/test.py configs/mmcls/classification_ncnn_static.py $MODEL_CONFIG --model reset18/end2end.param resnet18/end2end.bin --metrics accuracy precision f1_score recall
```

## 三、编译 SDK

### 1. 下载交叉编译工具链，设置环境变量

```bash
# 下载 Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.6-20220516.tar.gz
# https://occ.t-head.cn/community/download?id=4046947553902661632
$ tar xf Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.6-20220516.tar.gz
$ export RISCV_ROOT_PATH=`realpath Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.6`
```

### 2. 编译 ncnn & opencv

```bash
# ncnn
# 可参考 https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-allwinner-d1

# opencv
$ git clone https://github.com/opencv/opencv.git
$ mkdir build && cd build
$ cmake .. \
 -DCMAKE_TOOLCHAIN_FILE=/path/to/mmdeploy/cmake/toolchains/riscv64-linux-gnu.cmake \
 -DCMAKE_INSTALL_PREFIX=install \
 -DBUILD_PERF_TESTS=OFF \
 -DBUILD_SHARED_LIBS=OFF \
 -DBUILD_TESTS=OFF \
 -DCMAKE_BUILD_TYPE=Release
$ make -j$(nproc) && make install
```

### 3. 编译 mmdeploy SDK & demo

```bash
$ cd /path/to/mmdeploy
$ mkdir build && cd build
$ cmake .. \
    -DMMDEPLOY_BUILD_RISCV_SERVER=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/riscv64-linux-gnu.cmake \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DMMDEPLOY_SHARED_LIBS=OFF \
    -DMMDEPLOY_BUILD_EXAMPLES=ON \
    -DMMDEPLOY_TARGET_DEVICES="cpu" \
    -DMMDEPLOY_TARGET_BACKENDS="ncnn" \
    -Dncnn_DIR=/home/cx/ws/ncnn/build-c906/install/lib/cmake/ncnn/ \
    -DMMDEPLOY_CODEBASES=all \
    -DOpenCV_DIR=${OpenCV_DIR}/build/install/lib/cmake/opencv4

$ make -j$(nproc) && make install
$ tree -L 1 bin/
.
├── image_classification
├── image_restorer
├── image_segmentation
├── object_detection
├── ocr
├── pose_detection
└── rotated_object_detection
```

### 4. 运行 demo

先确认测试模型用了 `--dump-info`，这样 `resnet18` 目录才有 `pipeline.json` 等 SDK 所需文件。

把 dump 好的模型目录、可执行文件拷贝到设备中

```bash
./image_classification cpu ./resnet18  tiger.jpeg
```
