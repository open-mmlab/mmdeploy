## 如何转换模型

<!-- TOC -->

- [教程 : 如何转换模型](#如何转换模型)
    - [如何将模型从pytorch形式转换成其他后端形式](#如何将模型从pytorch形式转换成其他后端形式)
        - [准备工作](#准备工作)
        - [使用方法](#使用方法)
        - [参数描述](#参数描述)
    - [如何评测模型](#如何评测模型)
    - [各后端已支持导出的模型列表](#各后端已支持导出的模型列表)
    - [注意事项](#注意事项)
    - [问答](#问答)

<!-- TOC -->

这篇教程介绍了如何使用 MMDeploy 的工具将一个 OpenMMlab 模型转换成某个后端的模型文件。

注意:
- 现在已支持的后端包括 [ONNX Runtime](../backends/onnxruntime.md) ，[TensorRT](../backends/tensorrt.md) ，[NCNN](../backends/ncnn.md) ，[PPL](../backends/ppl.md)。
- 现在已支持的代码库包括 [MMClassification](../codebases/mmcls.md) ，[MMDetection](../codebases/mmdet.md) ，[MMSegmentation](../codebases/mmseg.md) ，[MMOCR](../codebases/mmocr.md) ，[MMEditing](../codebases/mmedit.md)。

### 如何将模型从pytorch形式转换成其他后端形式

#### 准备工作

1. 安装你的目标后端。 你可以参考 [ONNXRuntime-install](../backends/onnxruntime.md) ，[TensorRT-install](../backends/tensorrt.md) ，[NCNN-install](../backends/ncnn.md) ，[PPL-install](../backends/ppl.md)。
2. 安装你的目标代码库。 你可以参考 [MMClassification-install](https://github.com/open-mmlab/mmclassification/blob/master/docs/install.md) ，[MMDetection-install](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) ，[MMSegmentation-install](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation) ，[MMOCR-install](https://github.com/open-mmlab/mmocr/blob/main/docs/install.md) ，[MMEditing-install](https://github.com/open-mmlab/mmediting/blob/master/docs/install.md) 。

#### 使用方法

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

#### 参数描述

- `deploy_cfg` : MMDeploy 中用于部署的配置文件路径。
- `model_cfg` : OpenMMLab 系列代码库中使用的模型配置文件路径。
- `checkpoint` : OpenMMLab 系列代码库的模型文件路径。
- `img` : 用于模型转换时使用的图像文件路径。
- `--test-img` : 用于测试模型的图像文件路径。默认设置成`None`。
- `--work-dir` : 工作目录，用来保存日志和模型文件。
- `--calib-dataset-cfg` : 用于校准的配置文件。默认是`None`。
- `--device` : 用于模型转换的设备。 默认是`cpu`。
- `--log-level` : 设置日记的等级，选项包括`'CRITICAL'， 'FATAL'， 'ERROR'， 'WARN'， 'WARNING'， 'INFO'， 'DEBUG'， 'NOTSET'`。 默认是`INFO`。
- `--show` : 是否显示检测的结果。
- `--dump-info` : 是否输出 SDK 信息。

#### 示例

```bash
python ./tools/deploy.py \
    configs/mmdet/single-stage/single-stage_tensorrt_dynamic-320x320-1344x1344.py \
    $PATH_TO_MMDET/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    $PATH_TO_MMDET/checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco.pth \
    $PATH_TO_MMDET/demo/demo.jpg \
    --work-dir work_dir \
    --show
```

### 如何评测模型

你可以尝试去评测转换出来的模型 ，参考 [how_to_evaluate_a_model](./how_to_evaluate_a_model.md)。

### 各后端已支持导出的模型列表

下列列表显示了目前哪些模型是可以导出的以及其支持的后端。

|    模型       |     代码库        | 模型配置文件(示例)                                                                           | OnnxRuntime |    TensorRT   | NCNN |  PPL  |
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

### 注意事项

- None

### 问答

- None
