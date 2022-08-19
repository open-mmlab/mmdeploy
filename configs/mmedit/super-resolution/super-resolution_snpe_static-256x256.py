_base_ = ['./super-resolution_static.py', '../../_base_/backends/snpe.py']
onnx_config = dict(input_shape=[256, 256])
