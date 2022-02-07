## 学习如何配置文件

<!-- This tutorial describes how to write a config for model conversion and deployment. A deployment config includes `onnx config`, `codebase config`, `backend config`. -->
这篇教程介绍了如何编写模型转换和部署的配置文件。部署配置文件包括`onnx config`， `codebase config`， `backend config`。

<!-- TOC -->

- [如何编写配置文件](#如何编写配置文件)
  - [1. 如何编写ONNX配置文件](#1.如何编写ONNX配置文件)
    - [ONNX配置文件参数说明](#ONNX配置文件参数说明)
      - [示例](#示例)
    - [动态输入输出配置](#动态输入输出配置)
      - [示例](#示例-1)
  - [2. 如何编写OpenMMLab 系列代码库配置文件](#如何编写mmlab代码库类型配置文件)
    - [OpenMMLab 系列代码库配置文件参数说明](#OpenMMLab系列代码库配置文件参数说明)
      - [示例](#示例-2)
  - [3. 如何编写推理框架配置文件](#如何编写推理框架配置文件)
    - [示例](#示例-3)
  - [4. 以TensorRT为推理框架的图像分类部署配置文件示例](#4-a-complete-example-of-mmcls-on-tensorrt)
  - [5. 部署文件命名规则](#部署文件命名规则)
    - [示例](#示例-4)
  - [6. 如何编写模型配置文件](#如何编写模型配置文件)
  - [7. 注意事项](#注意事项)
  - [8. 常见问题](#常见问题)

<!-- TOC -->

### 1. 如何编写ONNX配置文件

ONNX 配置文件描述了如何将pytorch模型转换为ONNX模型。

#### ONNX配置文件参数说明

- `type`: 配置文件类型。 默认为 `onnx`。
- `export_params`: 如果指定，将导出模型所有参数。如果你只想导出未训练模型将此项设置为 False。
- `keep_initializers_as_inputs`: 
如果为 True，则所有初始化器（通常对应为参数）也将作为输入导出添加到计算图中。 如果为 False，则初始化器不会作为输入导出添加到计算图中，仅将非参数输入添加到计算图中。

- `opset_version`: ONNX的OP版本。默认为11.
- `save_file`: 输出ONNX模型文件。
- `input_names`: 模型计算图中输入节点的名称。
- `output_names`: 模型计算图中输出节点的名称。
- `input_shape`: 模型输入张量的高度和宽度。

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

#### 动态尺寸输入和输出配置

如果模型要求动态尺寸的输入和输出，你需要在ONNX配置文件中加入dynamic_axes配置。

- `dynamic_axes`: 描述输入和输出的维度信息。

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

### 2. 如何编写OpenMMLab 系列代码库配置文件

OpenMMLab 系列代码库配置部分包含代码库类型和任务类型等信息。

#### OpenMMLab 系列代码库配置文件参数说明

- `type`: OpenMMLab 系列模型代码库, 包括 `mmcls`, `mmdet`, `mmseg`, `mmocr`, `mmedit`.
- `task`: OpenMMLab 系列模型任务类型, 具体请参考 [OpenMMLab 系列模型任务列表](#list-of-tasks-in-all-codebases).

##### 示例

```python
codebase_config = dict(type='mmcls', task='Classification')
```

### 3. 如何编写推理框架配置文件

推理框架配置文件主要用于指定模型运行在哪个推理框架，并提供模型在推理框架运行时所需的信息, 具体参考 [ONNX Runtime](../backends/onnxruntime.md), [TensorRT](../backends/tensorrt.md), [NCNN](../backends/ncnn.md), [PPLNN](../backends/pplnn.md).

- `type`: 模型推理框架, 包括 `onnxruntime`, `ncnn`, `pplnn`, `tensorrt`, `openvino`.

#### 示例

```python
backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=False, max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 512, 1024],
                    opt_shape=[1, 3, 1024, 2048],
                    max_shape=[1, 3, 2048, 2048])))
    ])
```

### 4.  以TensorRT为推理框架的图像分类部署配置文件示例

这里我们提供了一个以TensorRT为推理框架的基于mmcls图像分类任务的完整部署配置文件示例。

```python

codebase_config = dict(type='mmcls', task='Classification')

backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=False,
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
```

### 5. 部署文件命名规则

这里我们事先约定一个特定的部署配置文件命名规则。


```bash
(task name)_(backend name)_(dynamic or static).py
```

- `task name`: 模型任务类型。
- `backend name`: 推理框架名称. 注意：如果你使用了量化，你需要指出量化类型。例如  `tensorrt-int8`。
- `dynamic or static`: 动态或者静态导出. 注意：如果推理框架需要明确的形状信息，您需要添加输入大小的描述，格式为`高度 x 宽度`。 例如 `dynamic-512x1024-2048x2048`, 这意味着最小输入形状是`512x1024`，最大输入形状是`2048x2048`。

#### 命名规则示例

```bash
detection_tensorrt-int8_dynamic-320x320-1344x1344.py
```

### 6. 如何编写模型配置文件

请根据模型具体任务的代码库，编写模型配置文件。 模型配置文件用于初始化模型，详情请参考[MMClassification](https://github.com/open-mmlab/mmclassification/blob/master/docs/zh_CN/tutorials/config.md), [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/zh_cn/tutorials/config.md), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/tutorials/config.md), [MMOCR](https://github.com/open-mmlab/mmocr/blob/main/docs/en/tutorials/config.md), [MMEditing](https://github.com/open-mmlab/mmediting/blob/master/docs/zh_cn/config.md).

### 7. 注意事项

None

### 8. 常见问题

None
