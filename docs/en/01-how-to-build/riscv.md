# Build for RISC-V

MMDeploy chooses ncnn as the inference backend on RISC-V platform. The deployment process consists of two steps:

Model conversion: Convert the PyTorch model to the ncnn model on the host side, and then upload the converted model to the device.

Model deployment: Compile ncnn and MMDeploy in cross-compilation mode on the host side, and then upload the executable for inference.

## 1. Model conversion

a) Install MMDeploy

You can refer to [Build document](./linux-x86_64.md) to install ncnn inference engine and MMDeploy

b) Convert model

```bash
export MODEL_CONFIG=/path/to/mmpretrain/configs/resnet/resnet18_8xb32_in1k.py
export MODEL_PATH=https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth

# Convert the model
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

## 2. Model deployment

a) Download the compiler toolchain and set environment

```bash
# download Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.6-20220516.tar.gz
# https://occ.t-head.cn/community/download?id=4046947553902661632
tar xf Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.6-20220516.tar.gz
export RISCV_ROOT_PATH=`realpath Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.6`
```

b) Compile ncnn & opencv

```bash
# ncnn
# refer to https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-allwinner-d1

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

c) Compile mmdeploy SDK & demo

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

After `make install`, the examples will locate in `install\bin`

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

### 4) Run the demo

First make sure that `--dump-info` is used during convert model, so that the `resnet18` directory has the files required by the SDK such as `pipeline.json`.

Copy the model folder(resnet18), executable(image_classification) file and test image(tests/data/tiger.jpeg) to the device.

```bash
./image_classification cpu ./resnet18  tiger.jpeg
```
