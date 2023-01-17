_base_ = ['./segmentation_dynamic.py', '../_base_/backends/openvino.py']

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 1024, 2048]))])
