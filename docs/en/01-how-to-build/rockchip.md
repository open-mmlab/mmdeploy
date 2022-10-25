# Build for RKNN

This tutorial is based on Linux systems like Ubuntu-18.04 and Rockchip NPU like `rk3588`.

## Installation

It is recommended to create a virtual environment for the project.

1. get RKNN-Toolkit2 through:

   ```
   git clone git@github.com:rockchip-linux/rknn-toolkit2.git
   ```

2. install RKNN python package following [official doc](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/doc). In our testing, we used the rknn-toolkit2 1.2.0 with commit id `834ba0b0a1ab8ee27024443d77b02b5ba48b67fc`. When installing rknn-toolkit2, it is better to append `--no-deps` after the commands to avoid dependency conflicts. For example:

   ```
   pip install packages/rknn_toolkit2-1.2.0_f7bb160f-cp36-cp36m-linux_x86_64.whl --no-deps
   ```

3. Install ONNX==1.8.0 before reinstall MMDeploy from source following the [instructions](../01-how-to-build/build_from_source.md). Note that there are conflicts between the pip dependencies of MMDeploy and RKNN. Here is the suggested packages versions for python 3.6:

   ```
   protobuf==3.19.4
   onnx==1.8.0
   onnxruntime==1.8.0
   torch==1.8.0
   torchvision==0.9.0
   ```

4. Install torch and torchvision using conda. For example:

```
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

To work with models from [MMClassification](https://mmclassification.readthedocs.io/en/latest/getting_started.html), you may need to install it additionally.

## Usage

Example:

```bash
python tools/deploy.py \
    configs/mmcls/classification_rknn_static.py \
    /mmclassification_dir/configs/resnet/resnet50_8xb32_in1k.py \
    https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth \
    /mmclassification_dir/demo/demo.JPEG \
    --work-dir ../resnet50 \
    --device cpu
```

## Deployment config

With the deployment config, you can modify the `backend_config` for your preference. An example `backend_config` of mmclassification is shown as below:

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

The contents of `common_config` are for `rknn.config()`. The contents of `quantization_config` are used to control `rknn.build()`.

## Build SDK with Rockchip NPU

1. get rknpu2 through:

   ```
   git clone git@github.com:rockchip-linux/rknpu2.git
   ```

2. for linux, download gcc cross compiler. The download link of the compiler from the official user guide of `rknpu2` was deprecated. You may use another verified [link](https://github.com/Caesar-github/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu). After download and unzip the compiler, you may open the terminal, set `RKNN_TOOL_CHAIN` and `RKNPU2_DEVICE_DIR` by `export RKNN_TOOL_CHAIN=/path/to/gcc/usr;export RKNPU2_DEVICE_DIR=/path/to/rknpu2/runtime/RK3588`.

3. after the above preparition, run the following commands:

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

## Run the demo with SDK

First make sure that`--dump-info`is used during convert model, so that the working directory has the files required by the SDK such as `pipeline.json`.

`adb push` the model directory, executable file and .so to the device.

```bash
cd /path/to/mmdeploy
adb push resnet50  /data/local/tmp/resnet50
adb push /mmclassification_dir/demo/demo.JPEG /data/local/tmp/resnet50/demo.JPEG
cd build
adb push lib /data/local/tmp/lib
adb push bin/image_classification /data/local/tmp/image_classification
```

Set up environment variable and execute the sample.

```bash
adb shell
cd /data/local/tmp
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/data/local/tmp/lib
./image_classification cpu ./resnet50  ./resnet50/demo.JPEG
..
label: 65, score: 0.95
```

## Troubleshooting

- Quantization fails.

  Empirically, RKNN require the inputs not normalized if `do_quantization` is set to `True`. Please modify the settings of `Normalize` in the `model_cfg` from

  ```python
  img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
  ```

  to

  ```python
  img_norm_cfg = dict(
    mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
  ```

  Besides, the `mean_values` and `std_values` of deploy_cfg should be replaced with original normalization settings of `model_cfg`. Let `mean_values=[123.675, 116.28, 103.53]` and `std_values=[58.395, 57.12, 57.375]`.
