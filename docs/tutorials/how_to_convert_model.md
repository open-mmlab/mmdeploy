## How to convert model

<!-- TOC -->

- [Tutorial : How to convert model](#how-to-convert-model)
    - [How to convert models from Pytorch to BACKEND](#how-to-convert-models-from-pytorch-to-other-backends)
        - [Prerequisite](#prerequisite)
        - [Usage](#usage)
        - [Description of all arguments](#description-of-all-arguments)
        - [How to find the corresponding deployment config of a PyTorch model](#how-to-find-the-corresponding-deployment-config-of-a-pytorch-model)
        - [Example](#example)
    - [How to evaluate the exported models](#how-to-evaluate-the-exported-models)
    - [List of supported models exportable to BACKEND](#list-of-supported-models-exportable-to-other-backends)
    - [Reminders](#reminders)
    - [FAQs](#faqs)

<!-- TOC -->

This tutorial briefly introduces how to export an OpenMMlab model to a specific backend using MMDeploy tools.
Notes:
- Supported backends are [ONNXRuntime](../backends/onnxruntime.md), [TensorRT](../backends/tensorrt.md), [NCNN](../backends/ncnn.md), [PPL](../backends/ppl.md), [OpenVINO](../backends/openvino.md).
- Supported codebases are [MMClassification](../codebases/mmcls.md), [MMDetection](../codebases/mmdet.md), [MMSegmentation](../codebases/mmseg.md), [MMOCR](../codebases/mmocr.md), [MMEditing](../codebases/mmedit.md).

### How to convert models from Pytorch to other backends

#### Prerequisite

1. Install and build your target backend. You could refer to [ONNXRuntime-install](../backends/onnxruntime.md), [TensorRT-install](../backends/tensorrt.md), [NCNN-install](../backends/ncnn.md), [PPL-install](../backends/ppl.md), [OpenVINO-install](../backends/openvino.md) for more information.
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

#### How to find the corresponding deployment config of a PyTorch model

1. Find model's codebase folder in `configs/ `. Example, convert a yolov3 model you need to find `configs/mmdet` folder.
2. Find model's task folder in `configs/codebase_folder/ `. Just like yolov3 model, you need to find `configs/mmdet/single-stage` folder.
3. Find deployment config file in `configs/codebase_folder/task_folder/ `. Just like deploy yolov3 model you can use `configs/mmdet/single-stage/single-stage_onnxruntime_dynamic.py`.

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

| Model              | codebase         | OnnxRuntime | TensorRT | NCNN | PPL | OpenVINO | model config file(example)                                                            |
|--------------------|------------------|:-----------:|:--------:|:----:|:---:|:--------:|:--------------------------------------------------------------------------------------|
| RetinaNet          | MMDetection      |      Y      |    Y     |  Y   |  Y  |    Y     | $MMDET_DIR/configs/retinanet/retinanet_r50_fpn_1x_coco.py                             |
| Faster R-CNN       | MMDetection      |      Y      |    Y     |  Y   |  Y  |    Y     | $MMDET_DIR/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py                         |
| YOLOv3             | MMDetection      |      Y      |    Y     |  Y   |  Y  |    Y     | $MMDET_DIR/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py                           |
| YOLOX              | MMDetection      |      Y      |    ?     |  ?   |  ?  |    Y     | $MMDET_DIR/configs/yolox/yolox_tiny_8x8_300e_coco.py                                  |
| FCOS               | MMDetection      |      Y      |    Y     |  Y   |  N  |    Y     | $MMDET_DIR/configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.                       |
| FSAF               | MMDetection      |      Y      |    Y     |  Y   |  Y  |    Y     | $MMDET_DIR/configs/fsaf/fsaf_r50_fpn_1x_coco.py                                       |
| Mask R-CNN         | MMDetection      |      Y      |    Y     |  N   |  Y  |    Y     | $MMDET_DIR/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py                             |
| SSD                | MMDetection      |      Y      |    Y     |  Y   |  Y  |    Y     | $MMDET_DIR/configs/ssd/ssd300_coco.py                                                 |
| FoveaBox           | MMDetection      |      Y      |    ?     |  ?   |  ?  |    Y     | $MMDET_DIR/configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py                              |
| ATSS               | MMDetection      |      Y      |    ?     |  ?   |  ?  |    Y     | $MMDET_DIR/configs/atss/atss_r50_fpn_1x_coco.py                                       |
| Cascade R-CNN      | MMDetection      |      Y      |    ?     |  ?   |  Y  |    Y     | $MMDET_DIR/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py                       |
| Cascade Mask R-CNN | MMDetection      |      Y      |    ?     |  ?   |  Y  |    Y     | $MMDET_DIR/configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py                  |
| ResNet             | MMClassification |      Y      |    Y     |  Y   |  Y  |    N     | $MMCLS_DIR/configs/resnet/resnet18_b32x8_imagenet.py                                  |
| ResNeXt            | MMClassification |      Y      |    Y     |  Y   |  Y  |    N     | $MMCLS_DIR/configs/resnext/resnext50_32x4d_b32x8_imagenet.py                          |
| SE-ResNet          | MMClassification |      Y      |    Y     |  Y   |  Y  |    N     | $MMCLS_DIR/configs/seresnet/seresnet50_b32x8_imagenet.py                              |
| MobileNetV2        | MMClassification |      Y      |    Y     |  Y   |  Y  |    N     | $MMCLS_DIR/configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py                        |
| ShuffleNetV1       | MMClassification |      Y      |    Y     |  Y   |  Y  |    N     | $MMCLS_DIR/configs/shufflenet_v1/shufflenet_v1_1x_b64x16_linearlr_bn_nowd_imagenet.py |
| ShuffleNetV2       | MMClassification |      Y      |    Y     |  Y   |  Y  |    N     | $MMCLS_DIR/configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py |
| FCN                | MMSegmentation   |      Y      |    Y     |  Y   |  Y  |    N     | $MMSEG_DIR/configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py                          |
| PSPNet             | MMSegmentation   |      Y      |    Y     |  N   |  Y  |    N     | $MMSEG_DIR/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py                    |
| DeepLabV3          | MMSegmentation   |      Y      |    Y     |  Y   |  Y  |    N     | $MMSEG_DIR/configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py              |
| DeepLabV3+         | MMSegmentation   |      Y      |    Y     |  Y   |  Y  |    N     | $MMSEG_DIR/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py      |
| SRCNN              | MMEditing        |      Y      |    Y     |  N   |  Y  |    N     | $MMSEG_DIR/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py                     |
| ESRGAN             | MMEditing        |      Y      |    Y     |  N   |  Y  |    N     | $MMSEG_DIR/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py         |
| DBNet              | MMOCR            |      Y      |    Y     |  Y   |  Y  |    N     | $MMOCR_DIR/configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py               |
| CRNN               | MMOCR            |      Y      |    Y     |  Y   |  N  |    N     | $MMOCR_DIR/configs/textrecog/tps/crnn_tps_academic_dataset.py                         |

### Reminders

- None

### FAQs

- None
