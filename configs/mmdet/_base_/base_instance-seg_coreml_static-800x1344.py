_base_ = ['./base_torchscript.py', '../../_base_/backends/coreml.py']

ir_config = dict(
    input_shape=(1344, 800), output_names=['dets', 'labels', 'masks'])

backend_config = dict(model_inputs=[
    dict(
        input_shapes=dict(
            input=dict(
                min_shape=[1, 3, 800, 1344],
                max_shape=[1, 3, 800, 1344],
                default_shape=[1, 3, 800, 1344])))
])

# Don't know if this is necessary
codebase_config = dict(post_processing=dict(export_postprocess_mask=False))
