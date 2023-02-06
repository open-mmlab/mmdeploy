_base_ = [
    '../_base_/base_instance-seg_static.py',
    '../../_base_/backends/onnxruntime.py'
]

# Notice: Do not set input_shape in onnx_config!
# This will result in an incorrect scale_factor!
# The input shape will be automatically inferred
# from the model's test_pipeline config.
onnx_config = dict(input_shape=None)
codebase_config = dict(post_processing=dict(export_postprocess_mask=True))
