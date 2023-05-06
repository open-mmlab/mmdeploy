# Fuse Transform（Experimental）

MMDeploy provides ability to fuse transform for acceleration in some cases.

When make inference with SDK, one can edit the pipeline.json to turn on the fuse option.

To bring the ability of fuse transform to MMDeploy, you can refer to the use of CVFusion.

## 1. Use CVFusion

There are two ways to use CVFusion, one is to use the pre-generated kernel code, the other is to generate the code yourself.

A）Use pre-generated kernel code

i) Download the kernel code from here，unzip it and copy the csrc folder to the mmdeploy root folder.

[elena_kernel-20220823.tar.gz](https://github.com/open-mmlab/mmdeploy/files/9399795/elena_kernel-20220823.tar.gz)

ii) Add option `-DMMDEPLOY_ELENA_FUSION=ON` when compile MMDeploy.

B) Generate kernel code by yourself

i) Compile CVFusion

```bash
$ git clone --recursive https://github.com/OpenComputeLab/CVFusion.git
$ cd CVFusion
$ bash build.sh
```

```
# add OpFuse to PATH
$ export PATH=`pwd`/build/examples/MMDeploy:$PATH
```

ii) Download algorithm codebase

```bash
$ tree -L 1 .
├── mmdeploy
├── mmpretrain
├── mmdetection
├── mmsegmentation
├── ...
```

iii) Generate kernel code

```bash
python tools/elena/extract_transform.py ..
# The generated code will be saved to csrc/preprocess/elena/{cpu_kernel}/{cuda_kernel}
```

iv) Add option `-DMMDEPLOY_ELENA_FUSION=ON` when compile MMDeploy.

## 2. Model conversion

Add `--dump-info` argument when convert a model, this will generate files that SDK needs.

```bash
$ export MODEL_CONFIG=/path/to/mmpretrain/configs/resnet/resnet18_8xb32_in1k.py
$ export MODEL_PATH=https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth

$ python tools/deploy.py \
    configs/mmpretrain/classification_onnxruntime_static.py \
    $MODEL_CONFIG \
    $MODEL_PATH \
    tests/data/tiger.jpeg \
    --work-dir resnet18 \
    --device cpu \
    --dump-info
```

## 3. Model Inference

If the model preprocess supports fusion, there will be a filed named `fuse_transform` in `pipeline.json`. It represents fusion switch and the default value `false` stands for off. One need to edit this filed to `true` to use the fuse option.
