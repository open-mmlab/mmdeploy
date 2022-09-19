_base_ = ['./text-detection_dynamic.py', '../../_base_/backends/sdk.py']

codebase_config = dict(model_type='sdk')

backend_config = dict(pipeline=[
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(type='PackTextDetInputs', meta_keys=['img_path', 'ori_shape'])
])
