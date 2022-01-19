_base_ = [
    './voxel-detection_static.py', '../../_base_/backends/onnxruntime.py'
]

onnx_config = dict(input_shape=None)
