# Copyright (c) OpenMMLab. All rights reserved.
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import \
    register_torch_op


@register_torch_op
def coreml_nms(context, node):
    """bind CoreML NMS op."""
    inputs = _get_inputs(context, node)
    boxes = inputs[0]
    scores = inputs[1]
    iou_threshold = inputs[2]
    score_threshold = inputs[3]
    max_boxes = inputs[4]
    results = mb.non_maximum_suppression(
        boxes=boxes,
        scores=scores,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        max_boxes=max_boxes)

    context.add(tuple(results), torch_name=node.outputs[0])
