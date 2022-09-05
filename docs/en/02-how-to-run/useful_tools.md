# Useful Tools

Apart from `deploy.py`, there are other useful tools under the `tools/` directory.

## torch2onnx

This tool can be used to convert PyTorch model from OpenMMLab to ONNX.

### Usage

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

### Description of all arguments

- `deploy_cfg` : The path of the deploy config file in MMDeploy codebase.
- `model_cfg` : The path of model config file in OpenMMLab codebase.
- `checkpoint` : The path of the model checkpoint file.
- `img` : The path of the image file used to convert the model.
- `--work-dir` : Directory to save output ONNX models Default is `./work-dir`.
- `--device` : The device used for conversion. If not specified, it will be set to `cpu`.
- `--log-level` : To set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`.

## extract

ONNX model with `Mark` nodes in it can be partitioned into multiple subgraphs. This tool can be used to extract the subgraph from the ONNX model.

### Usage

```bash
python tools/extract.py \
    ${INPUT_MODEL} \
    ${OUTPUT_MODEL} \
    --start ${PARITION_START} \
    --end ${PARITION_END} \
    --log-level INFO
```

### Description of all arguments

- `input_model` : The path of input ONNX model. The output ONNX model will be extracted from this model.
- `output_model` : The path of output ONNX model.
- `--start` : The start point of extracted model with format `<function_name>:<input/output>`. The `function_name` comes from the decorator `@mark`.
- `--end` : The end point of extracted model with format `<function_name>:<input/output>`. The `function_name` comes from the decorator `@mark`.
- `--log-level` : To set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`.

### Note

To support the model partition, you need to add Mark nodes in the ONNX model. The Mark node comes from the `@mark` decorator.
For example, if we have marked the `multiclass_nms` as below, we can set `end=multiclass_nms:input` to extract the subgraph before NMS.

```python
@mark('multiclass_nms', inputs=['boxes', 'scores'], outputs=['dets', 'labels'])
def multiclass_nms(*args, **kwargs):
    """Wrapper function for `_multiclass_nms`."""
```

## onnx2pplnn

This tool helps to convert an `ONNX` model to an `PPLNN` model.

### Usage

```bash
python tools/onnx2pplnn.py \
    ${ONNX_PATH} \
    ${OUTPUT_PATH} \
    --device cuda:0 \
    --opt-shapes [224,224] \
    --log-level INFO
```

### Description of all arguments

- `onnx_path`: The path of the `ONNX` model to convert.
- `output_path`: The converted `PPLNN` algorithm path in json format.
- `device`: The device of the model during conversion.
- `opt-shapes`: Optimal shapes for PPLNN optimization. The shape of each tensor should be wrap with "\[\]" or "()" and the shapes of tensors should be separated by ",".
- `--log-level`: To set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`.

## onnx2tensorrt

This tool can be used to convert ONNX to TensorRT engine.

### Usage

```bash
python tools/onnx2tensorrt.py \
    ${DEPLOY_CFG} \
    ${ONNX_PATH} \
    ${OUTPUT} \
    --device-id 0 \
    --log-level INFO \
    --calib-file /path/to/file
```

### Description of all arguments

- `deploy_cfg` : The path of the deploy config file in MMDeploy codebase.
- `onnx_path` : The ONNX model path to convert.
- `output` : The path of output TensorRT engine.
- `--device-id` : The device index, default to `0`.
- `--calib-file` : The calibration data used to calibrate engine to int8.
- `--log-level` : To set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`.

## onnx2ncnn

This tool helps to convert an `ONNX` model to an `ncnn` model.

### Usage

```bash
python tools/onnx2ncnn.py \
    ${ONNX_PATH} \
    ${NCNN_PARAM} \
    ${NCNN_BIN} \
    --log-level INFO
```

### Description of all arguments

- `onnx_path` : The path of the `ONNX` model to convert from.
- `output_param` : The converted `ncnn` param path.
- `output_bin` : The converted `ncnn` bin path.
- `--log-level` : To set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`.

## profiler

This tool helps to test latency of models with PyTorch, TensorRT and other backends. Note that the pre- and post-processing is excluded when computing inference latency.

### Usage

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

### Description of all arguments

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

### Example:

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

And the output look like this:

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
