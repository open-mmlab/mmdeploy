# Ubuntu Cross Build aarch64

mmdeploy chose ncnn as the inference backend for aarch64 embedded linux devices. There are two parts:

Host

- model conversion
- cross build SDK and demo for embedded devices

Device

- Run converted model

## 1. Model Convert on Host

Refer to the doc to install [mmdeploy](../01-how-to-build/) and [mmpretrain](https://github.com/open-mmlab/mmpretrain), and convert resnet18 for model package

```bash
export MODEL_CONFIG=/path/to/mmpretrain/configs/resnet/resnet18_8xb32_in1k.py
export MODEL_PATH=https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth

# Convert resnet18
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

## 2. Cross Build on Host

It is recommended to compile directly with the script

```bash
sh -x tools/scripts/ubuntu_cross_build_aarch64.sh
```

The following is the manual process corresponding to the script:

a) Install aarch64 build tools

```bash
sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

b) Cross build opencv and install to /tmp/ocv-aarch64

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

c) Cross build ncnn and install to /tmp/ncnn-aarch64

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

d) Cross build mmdeploy

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

## 3. Execute on Device

Make sure that `--dump-info` is used during model conversion, so that the `resnet18` directory contains the files required by the SDK such as `pipeline.json`.

Copy the model folder(resnet18), executable(image_classification) file, test image(tests/data/tiger.jpeg) and prebuilt OpenCV(/tmp/ocv-aarch64) to the device.

```bash
./image_classification cpu ./resnet18  tiger.jpeg
..
label: 292, score: 0.9261
label: 282, score: 0.0726
label: 290, score: 0.0008
label: 281, score: 0.0002
label: 340, score: 0.0001
```
