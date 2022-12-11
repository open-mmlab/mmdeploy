# How to get partitioned ONNX models

MMDeploy supports exporting PyTorch models to partitioned onnx models. With this feature, users can define their partition policy and get partitioned onnx models at ease. In this tutorial, we will briefly introduce how to support partition a model step by step. In the example, we would break YOLOV3 model into two parts and extract the first part without the post-processing (such as anchor generating and NMS) in the onnx model.

## Step 1: Mark inputs/outpupts

To support the model partition, we need to add `Mark` nodes in the ONNX model. This could be done with mmdeploy's `@mark` decorator. Note that to make the `mark` work, the marking operation should be included in a rewriting function.

At first, we would mark the model input, which could be done by marking the input tensor `img` in the `forward` method of `BaseDetector` class, which is the parent class of all detector classes. Thus we name this marking point as `detector_forward` and mark the inputs as `input`. Since there could be three outputs for detectors such as `Mask RCNN`, the outputs are marked as  `dets`, `labels`, and `masks`. The following code shows the idea of adding mark functions and calling the mark functions in the rewrite. For source code, you could refer to [mmdeploy/codebase/mmdet/models/detectors/base.py](https://github.com/open-mmlab/mmdeploy/blob/86a50e343a3a45d7bc2ba3256100accc4973e71d/mmdeploy/codebase/mmdet/models/detectors/base.py)

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

Then, we have to mark the output feature of `YOLOV3Head`, which is the input argument `pred_maps` in `get_bboxes` method of `YOLOV3Head` class. We could add a internal function to only mark the `pred_maps` inside [`yolov3_head__get_bboxes`](https://github.com/open-mmlab/mmdeploy/blob/86a50e343a3a45d7bc2ba3256100accc4973e71d/mmdeploy/codebase/mmdet/models/dense_heads/yolo_head.py#L14) function as following.

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

Note that `pred_maps` is a list of `Tensor` and it has three elements. Thus, three `Mark` nodes with op name as `pred_maps.0`, `pred_maps.1`, `pred_maps.2` would be added in the onnx model.

## Step 2: Add partition config

After marking necessary nodes that would be used to split the model, we could add a deployment config file `configs/mmdet/detection/yolov3_partition_onnxruntime_static.py`. If you are not familiar with how to write config, you could check [write_config.md](../02-how-to-run/write_config.md).

In the config file, we need to add `partition_config`. The key part is `partition_cfg`, which contains elements of dict that designates the start nodes and end nodes of each model segments. Since we only want to keep `YOLOV3` without post-processing, we could set the `start` as `['detector_forward:input']`, and `end` as `['yolo_head:input']`. Note that `start` and `end` can have multiple marks.

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
            end=['yolo_head:input'],  # [mark_name:input/output, ...]
            output_names=[f'pred_maps.{i}' for i in range(3)]) # output names
    ])

```

## Step 3: Get partitioned onnx models

Once we have marks of nodes and the deployment config with `parition_config` being set properly, we could use the [tool](../02-how-to-run/useful_tools.md) `torch2onnx` to export the model to onnx and get the partition onnx files.

```shell
python tools/torch2onnx.py \
configs/mmdet/detection/yolov3_partition_onnxruntime_static.py \
../mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth \
../mmdetection/demo/demo.jpg \
--work-dir ./work-dirs/mmdet/yolov3/ort/partition
```

After run the script above, we would have the partitioned onnx file `yolov3.onnx` in the `work-dir`. You can use the visualization tool [netron](https://netron.app/) to check the model structure.

With the partitioned onnx file, you could refer to [useful_tools.md](../02-how-to-run/useful_tools.md) to do the following procedures such as `mmdeploy_onnx2ncnn`, `onnx2tensorrt`.
