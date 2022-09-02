# 如何拆分 onnx 模型

MMDeploy 支持将PyTorch模型导出到onnx模型并进行拆分得到多个onnx模型文件，用户可以自由的对模型图节点进行标记并根据这些标记的节点定制任意的onnx模型拆分策略。在这个教程中，我们将通过具体例子来展示如何进行onnx模型拆分。在这个例子中，我们的目标是将YOLOV3模型拆分成两个部分，保留不带后处理的onnx模型，丢弃包含Anchor生成，NMS的后处理部分。

## 步骤 1: 添加模型标记点

为了进行图拆分，我们定义了`Mark`类型op，标记模型导出的边界。在实现方法上，采用`mark`装饰器对函数的输入、输出`Tensor`打标记。需要注意的是，我们的标记函数需要在某个重写函数中执行才能生效。

为了对YOLOV3进行拆分，首先我们需要标记模型的输入。这里为了通用性，我们标记检测器父类`BaseDetector`的`forward`方法中的`img` `Tensor`，同时为了支持其他拆分方案，也对`forward`函数的输出进行了标记，分别是`dets`, `labels`和`masks`。下面的代码是截图[mmdeploy/codebase/mmdet/models/detectors/base.py](https://github.com/open-mmlab/mmdeploy/blob/86a50e343a3a45d7bc2ba3256100accc4973e71d/mmdeploy/codebase/mmdet/models/detectors/base.py)中的一部分，可以看出我们使用`mark`装饰器标记了`__forward_impl`函数的输入输出，并在重写函数`base_detector__forward`进行了调用，从而完成了对检测器输入的标记。

```python
from mmdeploy.core import FUNCTION_REWRITER, mark

@mark(
    'detector_forward', inputs=['input'], outputs=['dets', 'labels', 'masks'])
def __forward_impl(ctx, self, img, img_metas=None, **kwargs):
    ...


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.base.BaseDetector.forward')
def base_detector__forward(ctx, self, img, img_metas=None, **kwargs):
    ...
    # call the mark function
    return __forward_impl(...)
```

接下来，我们只需要对`YOLOV3Head`中最后一层输出特征`Tensor`进行标记就可以将整个`YOLOV3`模型拆分成两部分。通过查看`mmdet`源码我们可以知道`YOLOV3Head`的`get_bboxes`方法中输入参数`pred_maps`就是我们想要的拆分点，因此可以在重写函数[`yolov3_head__get_bboxes`](https://github.com/open-mmlab/mmdeploy/blob/86a50e343a3a45d7bc2ba3256100accc4973e71d/mmdeploy/codebase/mmdet/models/dense_heads/yolo_head.py#L14)中添加内部函数对`pred_mapes`进行标记，具体参考如下示例代码。值得注意的是，输入参数`pred_maps`是由三个`Tensor`组成的列表，所以我们在onnx模型中添加了三个`Mark`标记节点。

```python
from mmdeploy.core import FUNCTION_REWRITER, mark

@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.YOLOV3Head.get_bboxes')
def yolov3_head__get_bboxes(ctx,
                            self,
                            pred_maps,
                            img_metas,
                            cfg=None,
                            rescale=False,
                            with_nms=True):
    # mark pred_maps
    @mark('yolo_head', inputs=['pred_maps'])
    def __mark_pred_maps(pred_maps):
        return pred_maps
    pred_maps = __mark_pred_maps(pred_maps)
    ...
```

## 步骤 2: 添加部署配置文件

在完成模型中节点标记之后，我们需要创建部署配置文件，我们假设部署后端是`onnxruntime`,并模型输入是固定尺寸`608x608`,因此添加文件`configs/mmdet/detection/yolov3_partition_onnxruntime_static.py`. 我们需要在配置文件中添加基本的配置信息如`onnx_config`，如何你还不熟悉如何添加配置文件，可以参考[write_config.md](../02-how-to-run/write_config.md).

在这个部署配置文件中, 我们需要添加一个特殊的模型分段配置字段`partition_config`. 在模型分段配置中，我们可以可以给分段策略添加一个类型名称如`yolov3_partition`，设定`apply_marks=True`。在分段方式`partition_cfg`,我们需要指定每段模型的分割起始点`start`, 终止点`end`以及保存分段onnx的文件名。需要提醒的是，各段模型起始点`start`和终止点`end`是由多个标记节点`Mark`组成，例如`'detector_forward:input'`代表`detector_forward`标记处输入所产生的标记节点。配置文件具体内容参考如下代码：

```python
_base_ = ['./detection_onnxruntime_static.py']

onnx_config = dict(input_shape=[608, 608])
partition_config = dict(
    type='yolov3_partition', # the partition policy name
    apply_marks=True, # should always be set to True
    partition_cfg=[
        dict(
            save_file='yolov3.onnx', # filename to save the partitioned onnx model
            start=['detector_forward:input'], # [mark_name:input/output, ...]
            end=['yolo_head:input'])  # [mark_name:input/output, ...]
    ])

```

## 步骤 3: 拆分onnx模型

添加好节点标记和部署配置文件，我们可以使用`tools/torch2onnx.py`工具导出带有`Mark`标记的完成onnx模型并根据分段策略提取分段的onnx模型文件。我们可以执行如下脚本，得到不带后处理的`YOLOV3`onnx模型文件`yolov3.onnx`，同时输出文件中也包含了添加`Mark`标记的完整模型文件`end2end.onnx`。此外，用户可以使用网页版模型可视化工具[netron](https://netron.app/)来查看和验证输出onnx模型的结构是否正确。

```shell
python tools/torch2onnx.py \
configs/mmdet/detection/yolov3_partition_onnxruntime_static.py \
../mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth \
../mmdetection/demo/demo.jpg \
--work-dir ./work-dirs/mmdet/yolov3/ort/partition
```

当得到分段onnx模型之后，我们可以使用mmdeploy提供的其他工具如`mmdeploy_onnx2ncnn`, `onnx2tensorrt`来进行后续的模型部署工作。
