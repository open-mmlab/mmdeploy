# RKNN support

This tutorial is based on Linux systems like Ubuntu-18.04 and Rockchip NPU like `rk3588`.

## Installation

It is recommended to create a virtual environment for the project.

1. get RKNN-Toolkit2 through:

   ```
   git clone https://github.com/rockchip-linux/rknn-toolkit2
   ```

2. install RKNN python package following [official doc](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/doc). In our testing, we used the rknn-toolkit 1.2.0 with commit id `834ba0b0a1ab8ee27024443d77b02b5ba48b67fc`.

3. reinstall MMDeploy from source following the [instructions](../01-how-to-build/build_from_source.md). Note that there are conflicts between the pip dependencies of MMDeploy and RKNN. Here is the suggested packages versions for python 3.6:

   ```
   protobuf==3.19.4
   onnx==1.8.0
   onnxruntime==1.8.0
   torch==1.8.0
   torchvision==0.9.0
   ```

To work with models from [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md), you may need to install it additionally.

## Usage

Example:

```bash
python tools/deploy.py \
    configs/mmdet/detection/detection_rknn_static.py \
    /mmdetection_dir/mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    /tmp/snapshots/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth \
    tests/data/tiger.jpeg \
    --work-dir ../deploy_result \
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

## Troubleshooting

- Quantization fails.

  Empirically, RKNN require the inputs not normalized if `do_quantization` is set to `False`. Please modify the settings of `Normalize` in the `model_cfg` from

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
