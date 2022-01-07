_base_ = ['../_base_/base_dynamic.py', '../../_base_/backends/sdk.py']

codebase_config = dict(model_type='sdk')

pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Collect', keys=['img'], meta_keys=['filename', 'ori_shape'])
]
