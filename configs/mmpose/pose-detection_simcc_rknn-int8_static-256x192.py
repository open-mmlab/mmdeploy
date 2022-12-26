_base_ = ['./pose-detection_static.py', '../_base_/backends/rknn.py']

onnx_config = dict(input_shape=[192, 256], output_names=['simcc_x', 'simcc_y'])

backend_config = dict(input_size_list=[[3, 256, 192]])
