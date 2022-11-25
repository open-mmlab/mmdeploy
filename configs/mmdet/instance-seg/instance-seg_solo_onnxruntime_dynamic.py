_base_ = [
    '../_base_/base_instance-seg_dynamic.py',
    '../../_base_/backends/onnxruntime.py'
]

codebase_config = dict(post_processing=dict(export_postprocess_mask=False))
