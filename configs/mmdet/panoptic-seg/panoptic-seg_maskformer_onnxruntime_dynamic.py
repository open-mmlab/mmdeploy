_base_ = [
    './panoptic-seg_maskformer_onnxruntime_static-800x1344.py',
]
onnx_config = dict(
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'cls_logits': {
            0: 'batch',
        },
        'mask_logits': {
            0: 'batch',
            2: 'h',
            3: 'w',
        },
    },
    input_shape=None)
