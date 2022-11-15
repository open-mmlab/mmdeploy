# Build for RKNN

This tutorial is based on Ubuntu-18.04 and Rockchip NPU `rk3588`. For different NPU devices, you may have to use different rknn packages.
Below is a table describing the relationship:

| Device               | Python Package                                                   | c/c++ SDK                                          |
| -------------------- | ---------------------------------------------------------------- | -------------------------------------------------- |
| RK1808/RK1806        | [rknn-toolkit](https://github.com/rockchip-linux/rknn-toolkit)   | [rknpu](https://github.com/rockchip-linux/rknpu)   |
| RV1109/RV1126        | [rknn-toolkit](https://github.com/rockchip-linux/rknn-toolkit)   | [rknpu](https://github.com/rockchip-linux/rknpu)   |
| RK3566/RK3568/RK3588 | [rknn-toolkit2](https://github.com/rockchip-linux/rknn-toolkit2) | [rknpu2](https://github.com/rockchip-linux/rknpu2) |
| RV1103/RV1106        | [rknn-toolkit2](https://github.com/rockchip-linux/rknn-toolkit2) | [rknpu2](https://github.com/rockchip-linux/rknpu2) |

## Installation

It is recommended to create a virtual environment for the project.

1. Get RKNN-Toolkit2 or RKNN-Toolkit through git. RKNN-Toolkit2 for example:

   ```
   git clone git@github.com:rockchip-linux/rknn-toolkit2.git
   ```

2. Install RKNN python package following [rknn-toolkit2 doc](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/doc) or [rknn-toolkit doc](https://github.com/rockchip-linux/rknn-toolkit/tree/master/doc). When installing rknn python package, it is better to append `--no-deps` after the commands to avoid dependency conflicts. RKNN-Toolkit2 package for example:

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

The contents of `common_config` are for `rknn.config()`. The contents of `quantization_config` are used to control `rknn.build()`. You may have to modify `target_platform` for your own preference.

## Build SDK with Rockchip NPU

### Build SDK with RKNPU2

1. Get rknpu2 through git:

   ```
   git clone git@github.com:rockchip-linux/rknpu2.git
   ```

2. For linux, download gcc cross compiler. The download link of the compiler from the official user guide of `rknpu2` was deprecated. You may use another verified [link](https://github.com/Caesar-github/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu). After download and unzip the compiler, you may open the terminal, set `RKNN_TOOL_CHAIN` and `RKNPU2_DEVICE_DIR` by `export RKNN_TOOL_CHAIN=/path/to/gcc/usr;export RKNPU2_DEVICE_DIR=/path/to/rknpu2/runtime/RK3588`.

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

- MMDet models.

  YOLOV3 & YOLOX: you may paste the following partition configuration into [detection_rknn_static.py](https://github.com/open-mmlab/mmdeploy/blob/master/configs/mmdet/detection/detection_rknn_static.py):

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

  RetinaNet & SSD & FSAF with rknn-toolkit2, you may paste the following partition configuration into [detection_rknn_static.py](https://github.com/open-mmlab/mmdeploy/blob/master/configs/mmdet/detection/detection_rknn_static.py). Users with rknn-toolkit can directly use default config.

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

- SDK only supports int8 rknn model, which require `do_quantization=True` when converting models.
