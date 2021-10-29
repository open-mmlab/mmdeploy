## 如何编写配置文件

这篇文档描述了如何去编写一个部署任务的配置文件。一次部署任务需要两个配置文件，一个是代码库模型的配置文件，另一个是部署任务的配置文件。一个部署任务的配置文件通常包括 `onnx config`， `codebase config`， `backend config`。

<!-- TOC -->

- [如何编写配置文件](#如何编写配置文件)
  - [1. 如何编写 onnx 配置](#1-如何编写-onnx-配置)
    - [onnx配置参数说明](#onnx配置参数说明)
      - [示例](#示例)
    - [如果你需要使用动态输入](#如果你需要使用动态输入)
      - [示例](#示例-1)
  - [2. 如何编写代码库的配置](#2-如何编写代码库的配置)
    - [代码库配置参数说明](#代码库配置参数说明)
      - [示例](#示例-2)
    - [如果你需要拆分模型](#如果你需要拆分模型)
      - [示例](#示例-3)
    - [各个代码库的任务类型列表](#各个代码库的任务类型列表)
  - [3. 如何编写后端的配置](#3-如何编写后端的配置)
    - [示例](#示例-4)
  - [4. 一个完整的mmcls模型部署在TensorRT的部署配置示例](#4-一个完整的mmcls模型部署在tensorrt的部署配置示例)
  - [5. 如何编写模型配置文件](#5-如何编写模型配置文件)
  - [6. 注意事项](#6-注意事项)
  - [7. 问答](#7-问答)

<!-- TOC -->

### 1. 如何编写 onnx 配置

onnx 的配置主要描述了模型由 pytorch 转换成 onnx 的过程中需要的信息。

#### onnx配置参数说明

- `type`： 表示该配置的类型。 默认是 `onnx`。
- `export_params`： 如果指定，将导出所有参数。 如果要导出未经训练的模型，请将其设置为 False。
- `keep_initializers_as_inputs`： 如果为 True，导出图中的所有初始值设定项（通常对应于参数）也将作为输入添加到图中。 如果为 False，则初始值设定项不会作为输入添加到图形中，而仅将非参数输入添加为输入。
- `opset_version`： 默认算子集版本是11。
- `save_file`： 指明输出的 onnx 文件。
- `input_names`： 表示 onnx 计算图的输入节点名。
- `output_names`： 表示 onnx 计算图的输出节点名。
- `input_shape`： 表示模型输入的尺度。

##### 示例

```python
onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['output'],
    input_shape=None)
```

#### 如果你需要使用动态输入

如果你需要使用动态输入作为输入和输出，你需要添加一个描述动态输入的配置在 onnx 的配置中。

- `dynamic_axes`： 描述动态输入的信息。

##### 示例

```python
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'dets': {
            0: 'batch',
            1: 'num_dets',
        },
        'labels': {
            0: 'batch',
            1: 'num_dets',
        },
    }
```

### 2. 如何编写代码库的配置

代码库的配置是描述模型的基础代码库和任务类型的。

#### 代码库配置参数说明

设置模型的基础代码库，包括

- `type`： 模型的基础代码库，包括 `mmcls`， `mmdet`， `mmseg`， `mmocr`， `mmedit`。
- `task`： 模型的任务类型，参考 [各个代码库的任务类型列表](#各个代码库的任务类型列表)。

##### 示例

```python
codebase_config = dict(type='mmcls', task='Classification')
```

#### 如果你需要拆分模型

如果你需要拆分模型，你需要添加一个拆分配置。注意，当前只有 MMDetection 的模型支持拆分。

- `type`： 设置模型任务， 参考[各个代码库的任务类型列表](#各个代码库的任务类型列表)。

##### 示例

```python
partition_config = dict(type='single_stage', apply_marks=True)
```

#### 各个代码库的任务类型列表

|      代码库       |        任务       | 是否可拆分  |
| :--------------: | :--------------: | :-------: |
| mmcls            | classification   |     N     |
| mmdet            | single-stage     |     Y     |
| mmdet            | two-stage        |     Y     |
| mmseg            | segmentation     |     N     |
| mmocr            | text-detection   |     N     |
| mmocr            | text-recognition |     N     |
| mmedit           | supe-resolution  |     N     |

### 3. 如何编写后端的配置

后端的配置主要是指定模型运行的后端和提供模型在后端上运行时需要的信息，参考 [ONNX Runtime](../backends/onnxruntime.md)，  [TensorRT](../backends/tensorrt.md)，  [NCNN](../backends/ncnn.md)，  [PPL](../backends/ppl.md)。

- `type`: 运行模型的后端，包括 `onnxruntime`， `ncnn`， `ppl`， `tensorrt`。

#### 示例

```python
backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=False, log_level=trt.Logger.INFO, max_workspace_size=1 << 30)
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 512, 1024],
                    opt_shape=[1, 3, 1024, 2048],
                    max_shape=[1, 3, 2048, 2048])))
    ])
```

### 4. 一个完整的mmcls模型部署在TensorRT的部署配置示例

这里我们展示一个完整的 mmcls 模型部署在 TensorRT 的部署配置示例。

```python
import tensorrt as trt

codebase_config = dict(type='mmcls', task='Classification')

backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=False,
        log_level=trt.Logger.INFO,
        max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 224, 224],
                    opt_shape=[4, 3, 224, 224],
                    max_shape=[64, 3, 224, 224])))])

onnx_config = dict(
    type='onnx',
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'output': {
            0: 'batch'
        }
    },
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['output'],
    input_shape=[224, 224])

partition_config = None
```

### 5. 如何编写模型配置文件

根据模型的代码库，编写模型配置文件。模型的配置文件用于初始化模型，参考 [MMClassification](https://github.com/open-mmlab/mmclassification/blob/master/docs_zh-CN/tutorials/config.md) ，[MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs_zh-CN/tutorials/config.md) ，[MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs_zh-CN/tutorials/config.md) ，[MMOCR](https://github.com/open-mmlab/mmocr/tree/main/configs) ，[MMEditing](https://github.com/open-mmlab/mmediting/blob/master/docs_zh-CN/config.md) 。

### 6. 注意事项

None

### 7. 问答

None
