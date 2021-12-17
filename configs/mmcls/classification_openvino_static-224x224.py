_base_ = ['./classification_static.py', '../_base_/backends/openvino.py']

onnx_config = dict(input_shape=(224, 224))
