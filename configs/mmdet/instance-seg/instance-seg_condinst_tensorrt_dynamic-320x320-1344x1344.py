_base_ = [
    '../_base_/base_instance-seg_dynamic.py',
    '../../_base_/backends/tensorrt.py'
]

# Notice: Do not set input_shape in onnx_config!
# This will result in an incorrect scale_factor!
# The input shape will be automatically inferred
# from the model's test_pipeline config.
onnx_config = dict(input_shape=None)
codebase_config = dict(post_processing=dict(export_postprocess_mask=True))
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 32),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 320, 320],
                    opt_shape=[1, 3, 800, 1216],
                    max_shape=[1, 3, 1344, 1344])))
    ])
