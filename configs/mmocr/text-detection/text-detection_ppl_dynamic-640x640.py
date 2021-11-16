_base_ = ['./text-detection_dynamic.py', '../../_base_/backends/ppl.py']

onnx_config = dict(input_shape=(640, 640))

backend_config = dict(model_inputs=dict(opt_shape=[1, 3, 640, 640]))
