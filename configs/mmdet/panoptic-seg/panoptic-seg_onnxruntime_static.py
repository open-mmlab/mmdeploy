_base_ = [
    '../_base_/base_panoptic-seg_static.py',
    '../../_base_/backends/onnxruntime.py'
]
onnx_config = dict(input_shape=[1280, 800])
