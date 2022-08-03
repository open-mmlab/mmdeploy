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


@register_torch_op
def log2(context, node):
    import math
    inputs = _get_inputs(context, node)
    x = inputs[0]
    log_x = mb.log(x=x)
    context.add(mb.mul(x=log_x, y=1 / math.log(2.0)), node.name)


# modify from:
# https://github.com/apple/coremltools/blob/bf5de6bfe6963d78fba68f471dc83be84f3d8c6e/coremltools/converters/mil/frontend/torch/ops.py
@register_torch_op
def roi_align(context, node):
    inputs = _get_inputs(context, node)

    x = context[node.inputs[0]]
    if len(x.shape) != 4:
        raise ValueError(
            '"CropResize" op: expected input rank 4, got {}'.format(x.rank))

    const_box_info = True
    if context[node.inputs[1]].val is None or context[
            node.inputs[2]].val is None:
        const_box_info = False

    extrapolation_value = context[node.inputs[2]].val

    # CoreML index information along with boxes
    if const_box_info:
        boxes = context[node.inputs[1]].val
        # CoreML expects boxes/ROI in
        # [N, 1, 5, 1, 1] format
        boxes = boxes.reshape(boxes.shape[0], 1, boxes.shape[1], 1, 1)
    else:
        boxes = inputs[1]
        boxes = mb.expand_dims(x=boxes, axes=[1, 3, 4])
    # Get Height and Width of crop
    h_out = inputs[3]
    w_out = inputs[4]

    # Crop Resize
    x = mb.crop_resize(
        x=x,
        roi=boxes,
        target_height=h_out.val,
        target_width=w_out.val,
        normalized_coordinates=False,
        spatial_scale=extrapolation_value,
        box_coordinate_mode='CORNERS_WIDTH_FIRST',
        sampling_mode='OFFSET_CORNERS',
    )

    # CoreML output format: [N, 1, C, h_out, w_out]
    # Torch output format: [N, C, h_out, w_out]
    x = mb.squeeze(x=x, axes=[1])

    context.add(x, torch_name=node.outputs[0])
