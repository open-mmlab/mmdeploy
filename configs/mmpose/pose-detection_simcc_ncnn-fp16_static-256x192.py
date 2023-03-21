_base_ = ['./pose-detection_static.py', '../_base_/backends/ncnn.py']

backend_config = dict(precision='FP16')
onnx_config = dict(input_shape=[192, 256], output_names=['simcc_x', 'simcc_y'])
