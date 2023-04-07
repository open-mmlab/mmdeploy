_base_ = ['./inpainting_static.py', '../../_base_/backends/onnxruntime.py']

onnx_config = dict(input_shape=[256, 256])
