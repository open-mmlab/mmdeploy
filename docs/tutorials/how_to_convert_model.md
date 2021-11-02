## How to convert model

<!-- TOC -->

- [Tutorial : How to convert model](#how-to-convert-model)
    - [How to convert models from Pytorch to BACKEND](#how-to-convert-models-from-pytorch-to-other-backends)
        - [Prerequisite](#prerequisite)
        - [Usage](#usage)
        - [Description of all arguments](#description-of-all-arguments)
    - [How to evaluate the exported models](#how-to-evaluate-the-exported-models)
    - [List of supported models exportable to BACKEND](#list-of-supported-models-exportable-to-other-backends)
    - [Reminders](#reminders)
    - [FAQs](#faqs)

<!-- TOC -->

This tutorial briefly introduces how to export an OpenMMlab model to a specific backend using MMDeploy tools.
Notes:
- Supported backends are [ONNXRuntime](../backends/onnxruntime.md), [TensorRT](../backends/tensorrt.md), [NCNN](../backends/ncnn.md), [PPL](../backends/ppl.md).
- Supported codebases are [MMClassification](../codebases/mmcls.md), [MMDetection](../codebases/mmdet.md), [MMSegmentation](../codebases/mmseg.md), [MMOCR](../codebases/mmocr.md), [MMEditing](../codebases/mmedit.md).

### How to convert models from Pytorch to other backends

#### Prerequisite

1. Install and build your target backend. You could refer to [ONNXRuntime-install](../backends/onnxruntime.md), [TensorRT-install](../backends/tensorrt.md), [NCNN-install](../backends/ncnn.md), [PPL-install](../backends/ppl.md) for more information.
2. Install and build your target codebase. You could refer to [MMClassification-install](https://github.com/open-mmlab/mmclassification/blob/master/docs/install.md), [MMDetection-install](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md), [MMSegmentation-install](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation), [MMOCR-install](https://github.com/open-mmlab/mmocr/blob/main/docs/install.md), [MMEditing-install](https://github.com/open-mmlab/mmediting/blob/master/docs/install.md).

#### Usage

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

#### Description of all arguments

- `deploy_cfg` : The path of deploy config file in MMDeploy codebase.
- `model_cfg` : The path of model config file in OpenMMLab codebase.
- `checkpoint` : The path of model checkpoint file.
- `img` : The path of image file that used to convert model.
- `--test-img` : The path of image file that used to test model. If not specified, it will be set to `None`.
- `--work-dir` : The path of work directory that used to save logs and models.
- `--calib-dataset-cfg` : Config used for calibration. If not specified, it will be set to `None`.
- `--device` : The device used for conversion. If not specified, it will be set to `cpu`.
- `--log-level` : To set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`.
- `--show` : Whether to show detection outputs.
- `--dump-info` : Whether to output information for SDK.

#### Example

```bash
python ./tools/deploy.py \
    configs/mmdet/single-stage/single-stage_tensorrt_dynamic-320x320-1344x1344.py \
    $PATH_TO_MMDET/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    $PATH_TO_MMDET/checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco.pth \
    $PATH_TO_MMDET/demo/demo.jpg \
    --work-dir work_dir \
    --show \
    --device cuda:0
```

### How to evaluate the exported models

You can try to evaluate model, referring to [how_to_evaluate_a_model](./how_to_evaluate_a_model.md).

### List of supported models exportable to other backends

The table below lists the models that are guaranteed to be exportable to other backend.

|    Model     |     codebase     | model config file(example)                                                                | OnnxRuntime |    TensorRT   | NCNN |  PPL  |
| :----------: | :--------------: | :---------------------------------------------------------------------------------------: | :---------: | :-----------: | :---:| :---: |
| RetinaNet    | MMDetection      | $PATH_TO_MMDET/configs/retinanet/retinanet_r50_fpn_1x_coco.py                             |      Y      |       Y       |   Y  |   Y   |
| Faster R-CNN | MMDetection      | $PATH_TO_MMDET/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py                         |      Y      |       Y       |   Y  |   Y   |
| YOLOv3       | MMDetection      | $PATH_TO_MMDET/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py                           |      Y      |       Y       |   N  |   Y   |
| FCOS         | MMDetection      | $PATH_TO_MMDET/configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py                     |      Y      |       Y       |   Y  |   N   |
| FSAF         | MMDetection      | $PATH_TO_MMDET/configs/fsaf/fsaf_r50_fpn_1x_coco.py                                       |      Y      |       Y       |   Y  |   Y   |
| Mask R-CNN   | MMDetection      | $PATH_TO_MMDET/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py                             |      Y      |       Y       |   N  |   Y   |
| ResNet       | MMClassification | $PATH_TO_MMCLS/configs/resnet/resnet18_b32x8_imagenet.py                                  |      Y      |       Y       |   Y  |   Y   |
| ResNeXt      | MMClassification | $PATH_TO_MMCLS/configs/resnext/resnext50_32x4d_b32x8_imagenet.py                          |      Y      |       Y       |   Y  |   Y   |
| SE-ResNet    | MMClassification | $PATH_TO_MMCLS/configs/seresnet/seresnet50_b32x8_imagenet.py                              |      Y      |       Y       |   Y  |   Y   |
| MobileNetV2  | MMClassification | $PATH_TO_MMCLS/configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py                        |      Y      |       Y       |   Y  |   Y   |
| ShuffleNetV1 | MMClassification | $PATH_TO_MMCLS/configs/shufflenet_v1/shufflenet_v1_1x_b64x16_linearlr_bn_nowd_imagenet.py |      Y      |       Y       |   N  |   Y   |
| ShuffleNetV2 | MMClassification | $PATH_TO_MMCLS/configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py |      Y      |       Y       |   N  |   Y   |
| FCN          | MMSegmentation   | $PATH_TO_MMSEG/configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py                          |      Y      |       Y       |   Y  |   Y   |
| PSPNet       | MMSegmentation   | $PATH_TO_MMSEG/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py                    |      Y      |       Y       |   N  |   Y   |
| DeepLabV3    | MMSegmentation   | $PATH_TO_MMSEG/configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py              |      Y      |       Y       |   Y  |   Y   |
| DeepLabV3+   | MMSegmentation   | $PATH_TO_MMSEG/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py      |      Y      |       Y       |   Y  |   Y   |
| SRCNN        | MMEditing        | $PATH_TO_MMSEG/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py                     |      Y      |       Y       |   N  |   Y   |
| ESRGAN       | MMEditing        | $PATH_TO_MMSEG/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py         |      Y      |       Y       |   N  |   Y   |
| DBNet        | MMOCR            | $PATH_TO_MMOCR/configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py               |      Y      |       Y       |   Y  |   Y   |
| CRNN         | MMOCR            | $PATH_TO_MMOCR/configs/textrecog/tps/crnn_tps_academic_dataset.py                         |      Y      |       Y       |   Y  |   N   |

### Reminders

- None

### FAQs

- None
