## MMDetection 支持

MMDetection 是一个基于 PyTorch 的目标检测开源工具箱。它是 [OpenMMLab](https://openmmlab.com/) 项目的一部分。

### MMDetection安装教程

请参考[快速入门文档](https://github.com/open-mmlab/mmdetection/blob/master/docs_zh-CN/get_started.md) 进行安装。

### MMDeploy支持的MMDetection模型列表

|    模型       |     任务类型      | 模型配置文件(示例)                                                                           | OnnxRuntime |    TensorRT   | NCNN |  PPLNN  |
| :----------: | :--------------: | :---------------------------------------------------------------------------------------: | :---------: | :-----------: | :---:| :---: |
| RetinaNet    | single-stage     | $PATH_TO_MMDET/configs/retinanet/retinanet_r50_fpn_1x_coco.py                             |      Y      |       Y       |   Y  |   Y   |
| Faster R-CNN | two-stage        | $PATH_TO_MMDET/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py                         |      Y      |       Y       |   Y  |   Y   |
| YOLOv3       | single-stage     | $PATH_TO_MMDET/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py                           |      Y      |       Y       |   N  |   Y   |
| FCOS         | single-stage     | $PATH_TO_MMDET/configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py                     |      Y      |       Y       |   Y  |   N   |
| FSAF         | single-stage     | $PATH_TO_MMDET/configs/fsaf/fsaf_r50_fpn_1x_coco.py                                       |      Y      |       Y       |   Y  |   Y   |
| Mask R-CNN   | two-stage        | $PATH_TO_MMDET/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py                             |      Y      |       Y       |   N  |   Y   |

### 注意事项

None

### 问答

None
