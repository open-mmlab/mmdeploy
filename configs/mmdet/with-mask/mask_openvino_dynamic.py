_base_ = [
    '../_base_/mask_base_dynamic.py', '../../_base_/backends/openvino.py'
]

codebase_config = dict(post_processing=dict(export_postprocess_mask=False))
