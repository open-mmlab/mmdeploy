# 量化测试结果

目前 mmdeploy 支持 ncnn 量化

## ncnn 量化

### 分类任务

|model|dataset|fp32 top-1 (%)|int8 top-1 (%)|
|:-:|:-:|:-:|:-:|
|[ResNet-18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb16_cifar10.py)|Cifar10|94.82|94.83|
|[ResNeXt-32x4d-50]()|ImageNet-1k|||

备注：分类模型量化后轻微（1% 以内）涨点或掉点是合理的

### OCR 检测任务

|model|dataset|fp32 hmean|int8 hmean|
|:-:|:-:|:-:|:-:|
|[PANet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py)|ICDAR2015|0.795|0.792 @thr=0.9|

备注：[mmocr](https://github.com/open-mmlab/mmocr) 使用 `shapely` 计算 IoU，实现方法会导致轻微的精度差异
