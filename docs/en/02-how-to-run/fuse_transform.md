# Fuse Transform（Experimental）

The prebuilt package for MMDeploy provide the ability to fuse transform for acceleration in some cases.

When make inference with SDK, one can edit the pipeline.json to turn on the fuse option.

## Model conversion

Add `--dump-info` argument when convert a model, this will generate files that SDK needs.

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

## Model Inference

If the model preprocess supports fusion, there will be a filed named `fuse_transform` in `pipeline.json`. It represents fusion switch and the default value `false` stands for off. One need to edit this filed to `true` to use the fuse option.
