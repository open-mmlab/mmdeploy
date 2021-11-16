_base_ = ['./classification_dynamic.py', '../_base_/backends/ppl.py']

onnx_config = dict(input_shape=[224, 224])

backend_config = dict(model_inputs=dict(opt_shape=[1, 3, 224, 224]))
