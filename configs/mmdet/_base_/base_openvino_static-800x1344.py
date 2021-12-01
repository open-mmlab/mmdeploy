_base_ = ['./base_static.py', '../../_base_/backends/openvino.py']

onnx_config = dict(input_shape=(1344, 800))
