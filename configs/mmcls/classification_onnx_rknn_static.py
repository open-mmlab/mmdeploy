_base_ = ['./classification_static.py', '../_base_/backends/rknn.py']

onnx_config = dict(input_shape=None)
backend_config = dict(
    # common_config=dict(mean_values=[0, 0, 0], std_values=[1, 1, 1]),
    input_size_list=[[1, 224, 224]])
