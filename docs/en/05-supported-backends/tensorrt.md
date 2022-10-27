# TensorRT Support

## Installation

### Install TensorRT

Please install TensorRT 8 follow [install-guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing).

**Note**:

- `pip Wheel File Installation` is not supported yet in this repo.

- We strongly suggest you install TensorRT through [tar file](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar)

- After installation, you'd better add TensorRT environment variables to bashrc by:

  ```bash
  cd ${TENSORRT_DIR} # To TensorRT root directory
  echo '# set env for TensorRT' >> ~/.bashrc
  echo "export TENSORRT_DIR=${TENSORRT_DIR}" >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$TENSORRT_DIR' >> ~/.bashrc
  source ~/.bashrc
  ```

### Build custom ops

Some custom ops are created to support models in OpenMMLab, and the custom ops can be built as follow:

```bash
cd ${MMDEPLOY_DIR} # To MMDeploy root directory
mkdir -p build && cd build
cmake -DMMDEPLOY_TARGET_BACKENDS=trt ..
make -j$(nproc)
```

If you haven't installed TensorRT in the default path, Please add `-DTENSORRT_DIR` flag in CMake.

```bash
 cmake -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=${TENSORRT_DIR} ..
 make -j$(nproc) && make install
```

## Convert model

Please follow the tutorial in [How to convert model](../02-how-to-run/convert_model.md). **Note** that the device must be `cuda` device.

### Int8 Support

Since TensorRT supports INT8 mode, a custom dataset config can be given to calibrate the model. Following is an example for MMDetection:

```python
# calibration_dataset.py

# dataset settings, same format as the codebase in OpenMMLab
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

Convert your model with this calibration dataset:

```python
python tools/deploy.py \
    ...
    --calib-dataset-cfg calibration_dataset.py
```

If the calibration dataset is not given, the data will be calibrated with the dataset in model config.

## FAQs

- Error `Cannot found TensorRT headers` or `Cannot found TensorRT libs`

  Try cmake with flag `-DTENSORRT_DIR`:

  ```bash
  cmake -DBUILD_TENSORRT_OPS=ON -DTENSORRT_DIR=${TENSORRT_DIR} ..
  make -j$(nproc)
  ```

  Please make sure there are libs and headers in `${TENSORRT_DIR}`.

- Error `error: parameter check failed at: engine.cpp::setBindingDimensions::1046, condition: profileMinDims.d[i] <= dimensions.d[i]`

  There is an input shape limit in deployment config:

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

  The shape of the tensor `input` must be limited between `input_shapes["input"]["min_shape"]` and `input_shapes["input"]["max_shape"]`.

- Error `error: [TensorRT] INTERNAL ERROR: Assertion failed: cublasStatus == CUBLAS_STATUS_SUCCESS`

  TRT 7.2.1 switches to use cuBLASLt (previously it was cuBLAS). cuBLASLt is the default choice for SM version >= 7.0. However, you may need CUDA-10.2 Patch 1 (Released Aug 26, 2020) to resolve some cuBLASLt issues. Another option is to use the new TacticSource API and disable cuBLASLt tactics if you don't want to upgrade.

  Read [this](https://forums.developer.nvidia.com/t/matrixmultiply-failed-on-tensorrt-7-2-1/158187/4) for detail.

- Install mmdeploy on Jetson

  We provide a tutorial to get start on Jetsons [here](../01-how-to-build/jetsons.md).
