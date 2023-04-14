# 支持 RISC-V

MMDeploy 选择 ncnn 作为 RISC-V 平台下的推理后端。完整的部署过程包含两个步骤：

模型转换：在 Host 端将 PyTorch 模型转为 ncnn 模型。并将转换后的模型传到 device。

模型部署：在 Host 端以交叉编译方式编译 ncnn 和 MMDeploy。传到 Device 端进行推理。

## 1. 模型转换

a) 安装 MMDeploy

可参考 [BUILD 文档](./linux-x86_64.md)，安装 ncnn 推理引擎以及 MMDeploy。

b) 模型转换

以 Resnet-18 为例。先参照[文档](https://github.com/open-mmlab/mmpretrain)安装 mmpretrain，然后使用 `tools/deploy.py` 转换模型。

```bash
export MODEL_CONFIG=/path/to/mmpretrain/configs/resnet/resnet18_8xb32_in1k.py
export MODEL_PATH=https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth

# 模型转换
cd /path/to/mmdeploy
python tools/deploy.py \
  configs/mmpretrain/classification_ncnn_static.py \
  $MODEL_CONFIG \
  $MODEL_PATH \
  tests/data/tiger.jpeg \
  --work-dir resnet18 \
  --device cpu \
  --dump-info
```

## 2. 模型部署

a) 下载交叉编译工具链，设置环境变量

```bash
# 下载 Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.6-20220516.tar.gz
# https://occ.t-head.cn/community/download?id=4046947553902661632
tar xf Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.6-20220516.tar.gz
export RISCV_ROOT_PATH=`realpath Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.6`
```

b) 编译 ncnn & opencv

```bash
# ncnn
# 可参考 https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-allwinner-d1

# opencv
git clone https://github.com/opencv/opencv.git
mkdir build_riscv && cd build_riscv
cmake .. \
 -DCMAKE_TOOLCHAIN_FILE=/path/to/mmdeploy/cmake/toolchains/riscv64-unknown-linux-gnu.cmake \
 -DCMAKE_INSTALL_PREFIX=install \
 -DBUILD_PERF_TESTS=OFF \
 -DBUILD_SHARED_LIBS=OFF \
 -DBUILD_TESTS=OFF \
 -DCMAKE_BUILD_TYPE=Release

make -j$(nproc) && make install
```

c) 编译 mmdeploy SDK & demo

```bash
cd /path/to/mmdeploy
mkdir build_riscv && cd build_riscv
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/riscv64-unknown-linux-gnu.cmake \
  -DMMDEPLOY_BUILD_SDK=ON \
  -DMMDEPLOY_SHARED_LIBS=OFF \
  -DMMDEPLOY_BUILD_EXAMPLES=ON \
  -DMMDEPLOY_TARGET_DEVICES="cpu" \
  -DMMDEPLOY_TARGET_BACKENDS="ncnn" \
  -Dncnn_DIR=${ncnn_DIR}/build-c906/install/lib/cmake/ncnn/ \
  -DMMDEPLOY_CODEBASES=all \
  -DOpenCV_DIR=${OpenCV_DIR}/build_riscv/install/lib/cmake/opencv4

make -j$(nproc) && make install
```

执行 `make install` 之后, examples的可执行文件会保存在 `install/bin`

```
tree -L 1 install/bin/
.
├── image_classification
├── image_restorer
├── image_segmentation
├── object_detection
├── ocr
├── pose_detection
└── rotated_object_detection
```

d) 运行 demo

先确认测试模型用了 `--dump-info`，这样 `resnet18` 目录才有 `pipeline.json` 等 SDK 所需文件。

把 dump 好的模型目录(resnet18)、可执行文件(image_classification)、测试图片(tests/data/tiger.jpeg)拷贝到设备中

```bash
./image_classification cpu ./resnet18  tiger.jpeg
```
