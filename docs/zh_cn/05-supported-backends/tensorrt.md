# TensorRT 支持情况

## 安装

### 安装TensorRT

请按照[安装指南](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing)安装TensorRT8。

**注意**:

- 此版本不支持`pip Wheel File Installation`。

- 我们强烈建议通过[tar包](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar)的方式安装TensorRT。

- 安装完成后，最好通过以下方式将TensorRT环境变量添加到bashrc:

  ```bash
  cd ${TENSORRT_DIR} # 进入TensorRT根目录
  echo '# set env for TensorRT' >> ~/.bashrc
  echo "export TENSORRT_DIR=${TENSORRT_DIR}" >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$TENSORRT_DIR' >> ~/.bashrc
  source ~/.bashrc
  ```

### 构建自定义算子

OpenMMLab中创建了一些自定义算子来支持模型，自定义算子可以如下构建:

```bash
cd ${MMDEPLOY_DIR} # 进入TensorRT根目录
mkdir -p build && cd build
cmake -DMMDEPLOY_TARGET_BACKENDS=trt ..
make -j$(nproc)
```

如果你没有在默认路径下安装TensorRT，请在CMake中添加`-DTENSORRT_DIR`标志。

```bash
 cmake -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=${TENSORRT_DIR} ..
 make -j$(nproc) && make install
```

## 转换模型

请遵循[如何转换模型](../02-how-to-run/convert_model.md)中的教程。**注意**设备必须是`cuda` 设备。

### Int8 支持

由于TensorRT支持INT8模式，因此可以提供自定义数据集配置来校准模型。MMDetection的示例如下:

```python
# calibration_dataset.py

# 数据集设置，格式与OpenMMLab中的代码库相同
dataset_type = 'CalibrationDataset'
data_root = 'calibration/dataset/root'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_annotations.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_annotations.json',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
```

使用此校准数据集转换您的模型:

```python
python tools/deploy.py \
    ...
    --calib-dataset-cfg calibration_dataset.py
```

如果没有提供校准数据集，则使用模型配置中的数据集进行校准。

## FAQs

- 错误 `Cannot found TensorRT headers`或`Cannot found TensorRT libs`

  可以尝试在cmake时使用`-DTENSORRT_DIR`标志:

  ```bash
  cmake -DBUILD_TENSORRT_OPS=ON -DTENSORRT_DIR=${TENSORRT_DIR} ..
  make -j$(nproc)
  ```

  请确保 `${TENSORRT_DIR}`中有库和头文件。

- 错误 `error: parameter check failed at: engine.cpp::setBindingDimensions::1046, condition: profileMinDims.d[i] <= dimensions.d[i]`

  在部署配置中有一个输入形状的限制:

  ```python
  backend_config = dict(
      # other configs
      model_inputs=[
          dict(
              input_shapes=dict(
                  input=dict(
                      min_shape=[1, 3, 320, 320],
                      opt_shape=[1, 3, 800, 1344],
                      max_shape=[1, 3, 1344, 1344])))
      ])
      # other configs
  ```

  `input` 张量的形状必须限制在`input_shapes["input"]["min_shape"]`和`input_shapes["input"]["max_shape"]`之间。

- 错误 `error: [TensorRT] INTERNAL ERROR: Assertion failed: cublasStatus == CUBLAS_STATUS_SUCCESS`

  TRT 7.2.1切换到使用cuBLASLt(以前是cuBLAS)。cuBLASLt是SM版本>= 7.0的默认选择。但是，您可能需要CUDA-10.2补丁1(2020年8月26日发布)来解决一些cuBLASLt问题。如果不想升级，另一个选择是使用新的TacticSource API并禁用cuBLASLt策略。

  请阅读[本文](https://forums.developer.nvidia.com/t/matrixmultiply-failed-on-tensorrt-7-2-1/158187/4)了解详情。

- 在Jetson上安装mmdeploy

  我们在[这里](../01-how-to-build/jetsons.md)提供了一个Jetsons入门教程。
