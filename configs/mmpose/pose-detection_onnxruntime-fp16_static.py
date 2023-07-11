_base_ = [
    './pose-detection_static.py', '../_base_/backends/onnxruntime-fp16.py'
]

onnx_config = dict(input_shape=None)
