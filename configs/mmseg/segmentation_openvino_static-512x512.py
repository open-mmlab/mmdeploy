_base_ = ['./segmentation_static.py', '../_base_/backends/openvino.py']

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 512, 512]))])
