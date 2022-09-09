_base_ = ['./segmentation_dynamic.py', '../_base_/backends/sdk.py']

codebase_config = dict(model_type='sdk')

backend_config = dict(pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='PackSegInputs', meta_keys=['img_path', 'ori_shape', 'img_shape'])
])
