# How to convert model

This tutorial briefly introduces how to export an OpenMMlab model to a specific backend using MMDeploy tools.
Notes:

- Supported backends are [ONNXRuntime](../05-supported-backends/onnxruntime.md), [TensorRT](../05-supported-backends/tensorrt.md), [ncnn](../05-supported-backends/ncnn.md), [PPLNN](../05-supported-backends/pplnn.md), [OpenVINO](../05-supported-backends/openvino.md).
- Supported codebases are [MMClassification](../04-supported-codebases/mmcls.md), [MMDetection](../04-supported-codebases/mmdet.md), [MMSegmentation](../04-supported-codebases/mmseg.md), [MMOCR](../04-supported-codebases/mmocr.md), [MMEditing](../04-supported-codebases/mmedit.md).

## How to convert models from Pytorch to other backends

### Prerequisite

1. Install and build your target backend. You could refer to [ONNXRuntime-install](../05-supported-backends/onnxruntime.md), [TensorRT-install](../05-supported-backends/tensorrt.md), [ncnn-install](../05-supported-backends/ncnn.md), [PPLNN-install](../05-supported-backends/pplnn.md), [OpenVINO-install](../05-supported-backends/openvino.md) for more information.
2. Install and build your target codebase. You could refer to [MMClassification-install](https://github.com/open-mmlab/mmclassification/blob/master/docs/en/install.md), [MMDetection-install](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md), [MMSegmentation-install](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation), [MMOCR-install](https://mmocr.readthedocs.io/en/latest/install.html), [MMEditing-install](https://github.com/open-mmlab/mmediting/blob/master/docs/en/install.md).

### Usage

```bash
python ./tools/deploy.py \
    ${DEPLOY_CFG_PATH} \
    ${MODEL_CFG_PATH} \
    ${MODEL_CHECKPOINT_PATH} \
    ${INPUT_IMG} \
    --test-img ${TEST_IMG} \
    --work-dir ${WORK_DIR} \
    --calib-dataset-cfg ${CALIB_DATA_CFG} \
    --device ${DEVICE} \
    --log-level INFO \
    --show \
    --dump-info
```

### Description of all arguments

- `deploy_cfg` : The path of deploy config file in MMDeploy codebase.
- `model_cfg` : The path of model config file in OpenMMLab codebase.
- `checkpoint` : The path of model checkpoint file.
- `img` : The path of image file that used to convert model.
- `--test-img` : The path of image file that used to test model. If not specified, it will be set to `None`.
- `--work-dir` : The path of work directory that used to save logs and models.
- `--calib-dataset-cfg` : Only valid in int8 mode. Config used for calibration. If not specified, it will be set to `None` and  use "val" dataset in model config for calibration.
- `--device` : The device used for conversion. If not specified, it will be set to `cpu`.
- `--log-level` : To set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`.
- `--show` : Whether to show detection outputs.
- `--dump-info` : Whether to output information for SDK.

### How to find the corresponding deployment config of a PyTorch model

1. Find model's codebase folder in `configs/`. Example, convert a yolov3 model you need to find `configs/mmdet` folder.
2. Find model's task folder in `configs/codebase_folder/`. Just like yolov3 model, you need to find `configs/mmdet/detection` folder.
3. Find deployment config file in `configs/codebase_folder/task_folder/`. Just like deploy yolov3 model you can use `configs/mmdet/detection/detection_onnxruntime_dynamic.py`.

### Example

```bash
python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    $PATH_TO_MMDET/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    $PATH_TO_MMDET/checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco.pth \
    $PATH_TO_MMDET/demo/demo.jpg \
    --work-dir work_dir \
    --show \
    --device cuda:0
```

## How to evaluate the exported models

You can try to evaluate model, referring to [how_to_evaluate_a_model](./how_to_evaluate_a_model.md).

## List of supported models exportable to other backends

Refer to [Support model list](../03-benchmark/supported_models.md)
