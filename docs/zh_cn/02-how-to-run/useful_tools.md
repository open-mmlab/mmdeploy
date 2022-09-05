# 更多工具介绍

除 `deploy.py` 以外， tools 目录下有很多实用工具

## torch2onnx

把 OpenMMLab 模型转 onnx 格式。

### 用法

```bash
python tools/torch2onnx.py \
    ${DEPLOY_CFG} \
    ${MODEL_CFG} \
    ${CHECKPOINT} \
    ${INPUT_IMG} \
    --work-dir ${WORK_DIR} \
    --device cpu \
    --log-level INFO
```

### 参数说明

- `deploy_cfg` : The path of the deploy config file in MMDeploy codebase.
- `model_cfg` : The path of model config file in OpenMMLab codebase.
- `checkpoint` : The path of the model checkpoint file.
- `img` : The path of the image file used to convert the model.
- `--work-dir` : Directory to save output ONNX models Default is `./work-dir`.
- `--device` : The device used for conversion. If not specified, it will be set to `cpu`.
- `--log-level` : To set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`.

## extract

有 `Mark` 节点的 onnx 模型会被分成多个子图，这个工具用来提取 onnx 模型中的子图。

### 用法

```bash
python tools/extract.py \
    ${INPUT_MODEL} \
    ${OUTPUT_MODEL} \
    --start ${PARITION_START} \
    --end ${PARITION_END} \
    --log-level INFO
```

### 参数说明

- `input_model` : The path of input ONNX model. The output ONNX model will be extracted from this model.
- `output_model` : The path of output ONNX model.
- `--start` : The start point of extracted model with format `<function_name>:<input/output>`. The `function_name` comes from the decorator `@mark`.
- `--end` : The end point of extracted model with format `<function_name>:<input/output>`. The `function_name` comes from the decorator `@mark`.
- `--log-level` : To set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`.

### 注意事项

要支持模型分块，必须在 onnx 模型中添加  mark 节点，用`@mark` 修饰。
下面这个例子里 mark 了 `multiclass_nms`，在 NMS 前设置 `end=multiclass_nms:input` 提取子图。

```python
@mark('multiclass_nms', inputs=['boxes', 'scores'], outputs=['dets', 'labels'])
def multiclass_nms(*args, **kwargs):
    """Wrapper function for `_multiclass_nms`."""
```

## onnx2pplnn

这个工具可以把 onnx 模型转成 pplnn 格式。

### 用法

```bash
python tools/onnx2pplnn.py \
    ${ONNX_PATH} \
    ${OUTPUT_PATH} \
    --device cuda:0 \
    --opt-shapes [224,224] \
    --log-level INFO
```

### 参数说明

- `onnx_path`: The path of the `ONNX` model to convert.
- `output_path`: The converted `PPLNN` algorithm path in json format.
- `device`: The device of the model during conversion.
- `opt-shapes`: Optimal shapes for PPLNN optimization. The shape of each tensor should be wrap with "\[\]" or "()" and the shapes of tensors should be separated by ",".
- `--log-level`: To set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`.

## onnx2tensorrt

这个工具把 onnx 转成 trt .engine 格式。

### 用法

```bash
python tools/onnx2tensorrt.py \
    ${DEPLOY_CFG} \
    ${ONNX_PATH} \
    ${OUTPUT} \
    --device-id 0 \
    --log-level INFO \
    --calib-file  /path/to/file
```

### 参数说明

- `deploy_cfg` : The path of the deploy config file in MMDeploy codebase.
- `onnx_path` : The ONNX model path to convert.
- `output` : The path of output TensorRT engine.
- `--device-id` : The device index, default to `0`.
- `--calib-file` : The calibration data used to calibrate engine to int8.
- `--log-level` : To set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`.

## onnx2ncnn

onnx 转 ncnn

### 用法

```bash
python tools/onnx2ncnn.py \
    ${ONNX_PATH} \
    ${NCNN_PARAM} \
    ${NCNN_BIN} \
    --log-level INFO
```

### 参数说明

- `onnx_path` : The path of the `ONNX` model to convert from.
- `output_param` : The converted `ncnn` param path.
- `output_bin` : The converted `ncnn` bin path.
- `--log-level` : To set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`.

## profiler

这个工具用来测试 torch 和 trt 等后端的速度，注意测试不包含前后处理。

### 用法

```bash
python tools/profiler.py \
    ${DEPLOY_CFG} \
    ${MODEL_CFG} \
    ${IMAGE_DIR} \
    --model ${MODEL} \
    --device ${DEVICE} \
    --shape ${SHAPE} \
    --num-iter ${NUM_ITER} \
    --warmup ${WARMUP} \
    --cfg-options ${CFG_OPTIONS} \
    --batch-size ${BATCH_SIZE} \
    --img-ext ${IMG_EXT}
```

### 参数说明

- `deploy_cfg` : The path of the deploy config file in MMDeploy codebase.
- `model_cfg` : The path of model config file in OpenMMLab codebase.
- `image_dir` : The directory to image files that used to test the model.
- `--model` : The path of the model to be tested.
- `--shape` : Input shape of the model by `HxW`, e.g., `800x1344`. If not specified, it would use `input_shape` from deploy config.
- `--num-iter` : Number of iteration to run inference. Default is `100`.
- `--warmup` : Number of iteration to warm-up the machine. Default is `10`.
- `--device` : The device type. If not specified, it will be set to `cuda:0`.
- `--cfg-options` : Optional key-value pairs to be overrode for model config.
- `--batch-size`: the batch size for test inference. Default is `1`. Note that not all models support `batch_size>1`.
- `--img-ext`: the file extensions for input images from `image_dir`. Defaults to `['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']`.

### 使用举例

```shell
python tools/profiler.py \
    configs/mmcls/classification_tensorrt_dynamic-224x224-224x224.py \
    ../mmclassification/configs/resnet/resnet18_8xb32_in1k.py \
    ../mmclassification/demo/ \
    --model work-dirs/mmcls/resnet/trt/end2end.engine \
    --device cuda \
    --shape 224x224 \
    --num-iter 100 \
    --warmup 10 \
    --batch-size 1
```

输出：

```text
----- Settings:
+------------+---------+
| batch size |    1    |
|   shape    | 224x224 |
| iterations |   100   |
|   warmup   |    10   |
+------------+---------+
----- Results:
+--------+------------+---------+
| Stats  | Latency/ms |   FPS   |
+--------+------------+---------+
|  Mean  |   1.535    | 651.656 |
| Median |   1.665    | 600.569 |
|  Min   |   1.308    | 764.341 |
|  Max   |   1.689    | 591.983 |
+--------+------------+---------+
```
