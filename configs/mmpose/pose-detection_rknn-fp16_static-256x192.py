_base_ = ['./pose-detection_static.py', '../_base_/backends/rknn.py']

onnx_config = dict(input_shape=[192, 256])

codebase_config = dict(model_type='end2end')

backend_config = dict(
    input_size_list=[[3, 256, 192]],
    quantization_config=dict(do_quantization=False),
    common_config=dict(target_platform='rv1126'))
