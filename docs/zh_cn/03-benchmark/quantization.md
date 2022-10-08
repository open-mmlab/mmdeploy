# 量化测试结果

目前 mmdeploy 支持 ncnn 量化

## ncnn 量化

### 分类任务

|                                                            model                                                             |   dataset   | fp32 top-1 (%) | int8 top-1 (%) |
| :--------------------------------------------------------------------------------------------------------------------------: | :---------: | :------------: | :------------: |
|       [ResNet-18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb16_cifar10.py)       |   Cifar10   |     94.82      |     94.83      |
| [ResNeXt-32x4d-50](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext50-32x4d_8xb32_in1k.py) | ImageNet-1k |     77.90      |    78.20\*     |
|  [MobileNet V2](https://github.com/open-mmlab/mmclassification/blob/master/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py)  | ImageNet-1k |     71.86      |    71.43\*     |
|       [HRNet-W18\*](https://github.com/open-mmlab/mmclassification/blob/master/configs/hrnet/hrnet-w18_4xb32_in1k.py)        | ImageNet-1k |     76.75      |    76.25\*     |

备注：

- 因为 imagenet-1k 数据量很大、ncnn 未正式发布 Vulkan int8 版本，考虑到 CPU 运行时间，仅用部分测试集（4000/50000）
- 量化后精度会有差异，分类模型涨点 1% 以内是正常情况

### OCR 检测任务

|                                                            model                                                             |  dataset  | fp32 hmean |   int8 hmean   |
| :--------------------------------------------------------------------------------------------------------------------------: | :-------: | :--------: | :------------: |
|      [PANet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py)       | ICDAR2015 |   0.795    | 0.792 @thr=0.9 |
| [TextSnake](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py) |  CTW1500  |   0.817    |     0.818      |

备注：[mmocr](https://github.com/open-mmlab/mmocr) 使用 `shapely` 计算 IoU，实现方法会导致轻微的精度差异

### 姿态检测任务

|                                                                      model                                                                       | dataset  | fp32 AP | int8 AP |
| :----------------------------------------------------------------------------------------------------------------------------------------------: | :------: | :-----: | :-----: |
| [Hourglass](https://github.com/open-mmlab/mmpose/blob/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hourglass52_coco_256x256.py) | COCO2017 |  0.726  |  0.713  |

备注：测试转换后的模型精度时，对于 mmpose 模型，在模型配置文件中 `flip_test` 需设置为 `False`。
