# Copyright (c) OpenMMLab. All rights reserved.
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import \
    register_torch_op


@register_torch_op(torch_alias=['mmdeploy::coreml_nms'])
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


@register_torch_op(override=True)
def stack(context, node):
    inputs = _get_inputs(context, node)

    values = inputs[0]

    if len(inputs) < 2:
        axis = 0
    else:
        axis = inputs[1]
        if hasattr(axis, 'val'):
            axis = axis.val
        if axis < 0:
            val_dim = len(values[0].shape)
            axis = axis + val_dim + 1

    res = mb.stack(values=values, axis=axis, name=node.name)
    context.add(res)


@register_torch_op(torch_alias=['torchvision::roi_align'])
def roi_align(context, node):
    """roi align."""
    inputs = _get_inputs(context, node)

    x = context[node.inputs[0]]
    input_shape = x.shape  # (B, C, h_in, w_in)
    if len(input_shape) != 4:
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
        boxes = mb.reshape(
            x=boxes, shape=[boxes.shape[0], 1, boxes.shape[1], 1, 1])
    # Get Height and Width of crop
    h_out = inputs[3]
    w_out = inputs[4]

    # Torch input format: [B, C, h_in, w_in]
    # CoreML input format: [B, C, h_in, w_in]

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
