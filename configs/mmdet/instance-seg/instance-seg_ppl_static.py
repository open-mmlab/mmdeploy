_base_ = [
    '../_base_/base_instance-seg_static.py', '../../_base_/backends/ppl.py'
]

codebase_config = dict(post_processing=dict(export_postprocess_mask=True))
