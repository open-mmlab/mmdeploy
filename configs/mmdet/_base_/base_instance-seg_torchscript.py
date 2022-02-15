_base_ = ['./base_torchscript.py']

ir_config = dict(
    output_names=['dets', 'labels', 'masks'], input_shape=[1344, 768])
codebase_config = dict(post_processing=dict(export_postprocess_mask=False))
