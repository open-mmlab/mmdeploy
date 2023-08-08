_base_ = [
    '../_base_/base_panoptic-seg_static.py',
    '../../_base_/backends/onnxruntime.py'
]
onnx_config = dict(
    input_shape=None,
    output_names=['dets', 'labels', 'masks', 'semseg'],
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
        'masks': {
            0: 'batch',
            1: 'num_dets',
        },
        'semseg': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
    },
)
