_base_ = ['../_base_/mask_base_static.py', '../../_base_/backends/ppl.py']

codebase_config = dict(post_processing=dict(export_postprocess_mask=True))
