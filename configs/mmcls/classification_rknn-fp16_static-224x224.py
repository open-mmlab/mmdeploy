_base_ = ['./classification_static.py', '../_base_/backends/rknn.py']

onnx_config = dict(input_shape=[224, 224])
codebase_config = dict(model_type='end2end')
backend_config = dict(
    input_size_list=[[3, 224, 224]],
    quantization_config=dict(do_quantization=False))
