# How to convert model

This tutorial briefly introduces how to export an OpenMMlab model to a specific backend using MMDeploy tools.
Notes:

- Supported backends are [ONNXRuntime](../05-supported-backends/onnxruntime.md), [TensorRT](../05-supported-backends/tensorrt.md), [ncnn](../05-supported-backends/ncnn.md), [PPLNN](../05-supported-backends/pplnn.md), [OpenVINO](../05-supported-backends/openvino.md).
- Supported codebases are [MMPretrain](../04-supported-codebases/mmpretrain.md), [MMDetection](../04-supported-codebases/mmdet.md), [MMSegmentation](../04-supported-codebases/mmseg.md), [MMOCR](../04-supported-codebases/mmocr.md), [MMagic](../04-supported-codebases/mmagic.md).

## How to convert models from Pytorch to other backends

### Prerequisite

1. Install and build your target backend. You could refer to [ONNXRuntime-install](../05-supported-backends/onnxruntime.md), [TensorRT-install](../05-supported-backends/tensorrt.md), [ncnn-install](../05-supported-backends/ncnn.md), [PPLNN-install](../05-supported-backends/pplnn.md), [OpenVINO-install](../05-supported-backends/openvino.md) for more information.
2. Install and build your target codebase. You could refer to [MMPretrain-install](https://mmpretrain.readthedocs.io/en/latest/get_started.html#installation), [MMDetection-install](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation), [MMSegmentation-install](https://mmsegmentation.readthedocs.io/en/latest/get_started.html#installation), [MMOCR-install](https://mmocr.readthedocs.io/en/latest/get_started/install.html#installation-steps), [MMagic-install](https://mmagic.readthedocs.io/en/latest/get_started/install.html#installation).

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

- `deploy_cfg` : The deployment configuration of mmdeploy for the model, including the type of inference framework, whether quantize, whether the input shape is dynamic, etc. There may be a reference relationship between configuration files, `mmdeploy/mmpretrain/classification_ncnn_static.py` is an example.
- `model_cfg` : Model configuration for algorithm library, e.g. `mmpretrain/configs/vision_transformer/vit-base-p32_ft-64xb64_in1k-384.py`, regardless of the path to mmdeploy.
- `checkpoint` : torch model path. It can start with http/https, see the implementation of `mmcv.FileClient` for details.
- `img` : The path to the image or point cloud file used for testing during the model conversion.
- `--test-img` : The path of the image file that is used to test the model. If not specified, it will be set to `None`.
- `--work-dir` : The path of the work directory that is used to save logs and models.
- `--calib-dataset-cfg` : Only valid in int8 mode. The config used for calibration. If not specified, it will be set to `None` and use the "val" dataset in the model config for calibration.
- `--device` : The device used for model conversion. If not specified, it will be set to `cpu`. For trt, use `cuda:0` format.
- `--log-level` : To set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`.
- `--show` : Whether to show detection outputs.
- `--dump-info` : Whether to output information for SDK.

### How to find the corresponding deployment config of a PyTorch model

1. Find the model's codebase folder in `configs/`.  For converting a yolov3 model, you need to check `configs/mmdet` folder.
2. Find the model's task folder in `configs/codebase_folder/`. For a yolov3 model, you need to check `configs/mmdet/detection` folder.
3. Find the deployment config file in `configs/codebase_folder/task_folder/`. For deploying a yolov3 model to the onnx backend, you could use `configs/mmdet/detection/detection_onnxruntime_dynamic.py`.

### Example

```bash
python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    $PATH_TO_MMDET/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py \
    $PATH_TO_MMDET/checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth \
    $PATH_TO_MMDET/demo/demo.jpg \
    --work-dir work_dir \
    --show \
    --device cuda:0
```

## How to evaluate the exported models

You can try to evaluate model, referring to [how_to_evaluate_a_model](profile_model.md).

## List of supported models exportable to other backends

Refer to [Support model list](../03-benchmark/supported_models.md)
