_base_ = [
    '../_base_/base_panoptic-seg_static.py',
    '../../_base_/backends/onnxruntime.py'
]
onnx_config = dict(
    opset_version=13,
    output_names=['cls_logits', 'mask_logits'],
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'cls_logits': {
            0: 'batch',
            2: 'h',
            3: 'w',
        },
        'mask_logits': {
            0: 'batch',
            2: 'h',
            3: 'w',
        },
    },
    input_shape=None)
