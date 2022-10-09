# 如何转换模型

<!-- TOC -->

- [如何转换模型](#如何转换模型)
  - [如何将模型从pytorch形式转换成其他后端形式](#如何将模型从pytorch形式转换成其他后端形式)
    - [准备工作](#准备工作)
    - [使用方法](#使用方法)
    - [参数描述](#参数描述)
    - [如何查找pytorch模型对应的部署配置文件](#如何查找pytorch模型对应的部署配置文件)
    - [示例](#示例)
  - [如何评测模型](#如何评测模型)
  - [各后端已支持导出的模型列表](#各后端已支持导出的模型列表)

<!-- TOC -->

这篇教程介绍了如何使用 MMDeploy 的工具将一个 OpenMMlab 模型转换成某个后端的模型文件。

注意:

- 现在已支持的后端包括 [ONNXRuntime](../05-supported-backends/onnxruntime.md) ，[TensorRT](../05-supported-backends/tensorrt.md) ，[ncnn](../05-supported-backends/ncnn.md) ，[PPLNN](../05-supported-backends/pplnn.md) ，[OpenVINO](../05-supported-backends/openvino.md)。
- 现在已支持的代码库包括 [MMClassification](../04-supported-codebases/mmcls.md) ，[MMDetection](../04-supported-codebases/mmdet.md) ，[MMSegmentation](../04-supported-codebases/mmseg.md) ，[MMOCR](../04-supported-codebases/mmocr.md) ，[MMEditing](../04-supported-codebases/mmedit.md)。

## 如何将模型从pytorch形式转换成其他后端形式

### 准备工作

1. 安装您的目标后端。 您可以参考 [ONNXRuntime-install](../05-supported-backends/onnxruntime.m) ，[TensorRT-install](../05-supported-backends/tensorrt.md) ，[ncnn-install](../05-supported-backends/ncnn.md) ，[PPLNN-install](../05-supported-backends/pplnn.md), [OpenVINO-install](../05-supported-backends/openvino.md)。
2. 安装您的目标代码库。 您可以参考 [MMClassification-install](https://github.com/open-mmlab/mmclassification/blob/master/docs/zh_CN/install.md)， [MMDetection-install](https://github.com/open-mmlab/mmdetection/blob/master/docs/zh_cn/get_started.md)， [MMSegmentation-install](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/get_started.md#installation)， [MMOCR-install](https://mmocr.readthedocs.io/zh_CN/latest/install.html)， [MMEditing-install](https://github.com/open-mmlab/mmediting/blob/master/docs/zh_cn/install.md)。

### 使用方法

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

### 参数描述

- `deploy_cfg` : mmdeploy 针对此模型的部署配置，包含推理框架类型、是否量化、输入 shape 是否动态等。配置文件之间可能有引用关系，`mmdeploy/mmcls/classification_ncnn_static.py` 是一个示例。
- `model_cfg` : mm 算法库的模型配置，例如 `mmclassification/configs/vision_transformer/vit-base-p32_ft-64xb64_in1k-384.py`，与 mmdeploy 的路径无关。
- `checkpoint` : torch 模型路径。可以 http/https 开头，详见 `mmcv.FileClient`   的实现。
- `img` : 模型转换时，用做测试的图像或点云文件路径。
- `--test-img` : 用于测试模型的图像文件路径。默认设置成`None`。
- `--work-dir` : 工作目录，用来保存日志和模型文件。
- `--calib-dataset-cfg` : 此参数只有int8模式下生效，用于校准数据集配置文件。若在int8模式下未传入参数，则会自动使用模型配置文件中的'val'数据集进行校准。
- `--device` : 用于模型转换的设备。 默认是`cpu`，对于 trt 可使用 `cuda:0` 这种形式。
- `--log-level` : 设置日记的等级，选项包括`'CRITICAL'， 'FATAL'， 'ERROR'， 'WARN'， 'WARNING'， 'INFO'， 'DEBUG'， 'NOTSET'`。 默认是`INFO`。
- `--show` : 是否显示检测的结果。
- `--dump-info` : 是否输出 SDK 信息。

### 如何查找pytorch模型对应的部署配置文件

1. 在 `configs/` 文件夹中找到模型对应的代码库文件夹。 例如，转换一个yolov3模型您可以查找到 `configs/mmdet` 文件夹。
2. 根据模型的任务类型在 `configs/codebase_folder/` 下查找对应的文件夹。 例如yolov3模型，您可以查找到 `configs/mmdet/detection` 文件夹。
3. 在 `configs/codebase_folder/task_folder/` 下找到模型的部署配置文件。 例如部署yolov3您可以使用 `configs/mmdet/detection/detection_onnxruntime_dynamic.py`。

### 示例

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

## 如何评测模型

您可以尝试去评测转换出来的模型 ，参考 [profile 模型](profile_model.md)。

## 各后端已支持导出的模型列表

参考[已支持的模型列表](../03-benchmark/supported_models.md)。
