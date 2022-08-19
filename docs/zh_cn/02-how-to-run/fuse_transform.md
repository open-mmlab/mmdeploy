# 融合预处理（实验性功能）

MMDeploy预编译包中的库集成了一些Transform融合的能力，当使用SDK进行推理时，可以通过修改pipeline.json来开启融合选项，在某些Transform的组合下可以对预处理进行加速。

## 模型转换

模型转换时通过`--dump-info`生成SDK所需文件。

```bash
$ export MODEL_CONFIG=/path/to/mmclassification/configs/resnet/resnet18_8xb32_in1k.py
$ export MODEL_PATH=https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth

$ python tools/deploy.py \
    configs/mmcls/classification_onnxruntime_static.py \
    $MODEL_CONFIG \
    $MODEL_PATH \
    /path/to/test.png \
    --work-dir resnet18 \
    --device cpu \
    --dump-info

```

## 模型推理

若当前pipeline的预处理模块支持融合，`pipeline.json`中会有`fuse_transform`字段，表示融合开关，默认为`false`。当启用融合算法时，需要把`false`改为`true`
