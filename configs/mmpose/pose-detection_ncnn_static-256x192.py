_base_ = ['./pose-detection_static.py', '../_base_/backends/ncnn.py']

onnx_config = dict(input_shape=[192, 256])
