_base_ = [
    '../_base_/base_instance-seg_dynamic.py',
    '../../_base_/backends/openvino.py'
]

onnx_config = dict(input_shape=None)
backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 300, 300]))])
codebase_config = dict(post_processing=dict(export_postprocess_mask=True))
