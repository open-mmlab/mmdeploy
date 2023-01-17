_base_ = ['./super-resolution_dynamic.py', '../../_base_/backends/openvino.py']

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 256, 256]))])
