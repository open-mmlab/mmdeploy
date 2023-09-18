_base_ = [
    '../_base_/base_instance-seg_dynamic.py',
    '../../_base_/backends/onnxruntime.py'
]

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
    ]
)
