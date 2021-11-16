_base_ = ['./segmentation_dynamic.py', '../_base_/backends/ppl.py']

onnx_config = dict(input_shape=[1024, 512])

backend_config = dict(model_inputs=dict(opt_shape=[1, 3, 512, 1024]))
