# 融合预处理（实验性功能）

MMDeploy提供了一些Transform融合的能力，当使用SDK进行推理时，可以通过修改pipeline.json来开启融合选项，在某些Transform的组合下可以对预处理进行加速。

若要在MMDeploy的SDK中加入融合能力，可参考CVFusion的使用。

## 一、使用CVFusion

有两种选择，一种是在编译mmdeploy的时候，使用我们提供的融合kernel代码，一种是自己使用CVFusion生成融合kernel的代码。

A）使用提供的kernel代码

1. 从这里下载代码，并解压，将csrc文件夹拷贝到mmdeploy的根目录。

   [elena_kernel-20220823.tar.gz](https://github.com/open-mmlab/mmdeploy/files/9399795/elena_kernel-20220823.tar.gz)

2. 编译mmdeploy的时候，增加选项`-DMMDEPLOY_ELENA_FUSION=ON`

B) 使用CVFusion生成kernel

1. 编译CVFusion

   ```bash
   $ git clone --recursive https://github.com/OpenComputeLab/CVFusion.git
   $ cd CVFusion
   $ bash build.sh

   # add OpFuse to PATH
   $ export PATH=`pwd`/build/examples/MMDeploy:$PATH
   ```

2. 下载各个算法codebase

   ```bash
   $ tree -L 1 .
   ├── mmdeploy
   ├── mmpretrain
   ├── mmdetection
   ├── mmsegmentation
   ├── ...
   ```

3. 生成融合kernel

   ```bash
   python tools/elena/extract_transform.py ..
   # 生成的代码会保存在csrc/preprocess/elena/{cpu_kernel}/{cuda_kernel}
   ```

4. 编译mmdeploy的时候，增加选项`-DMMDEPLOY_ELENA_FUSION=ON`

## 二、模型转换

模型转换时通过`--dump-info`生成SDK所需文件。

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

## 三、模型推理

若当前pipeline的预处理模块支持融合，`pipeline.json`中会有`fuse_transform`字段，表示融合开关，默认为`false`。当启用融合算法时，需要把`false`改为`true`
