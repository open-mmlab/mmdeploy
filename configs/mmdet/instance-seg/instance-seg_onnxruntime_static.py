_base_ = [
    '../_base_/base_instance-seg_static.py',
    '../../_base_/backends/onnxruntime.py'
]

onnx_config = dict(input_shape=(640, 640))
