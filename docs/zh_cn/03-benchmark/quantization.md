# 量化测试结果

目前 mmdeploy 支持 ncnn 量化

## ncnn 量化

### 分类任务

|model|dataset|fp32 top-1 (%)|int8 top-1 (%)|
|:-:|:-:|:-:|:-:|
|[ResNet-18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb16_cifar10.py)|Cifar10|94.82|94.83|
|[ResNeXt-32x4d-50]()|ImageNet-1k|||

### 检测任务
