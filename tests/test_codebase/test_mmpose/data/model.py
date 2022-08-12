# Copyright (c) OpenMMLab. All rights reserved.
# model settings
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)
test_cfg = dict(
    flip_test=False,
    flip_mode='heatmap',
    shift_heatmap=True,
)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(type='ResNet', depth=18),
    head=dict(
        type='HeatmapHead',
        in_channels=512,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=test_cfg)

# dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'tests/test_codebase/test_mmpose/data/'
file_client_args = dict(backend='disk')

test_pipeline = [
    dict(type='LoadImage', file_client_args=file_client_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        lazy_init=True,
        serialize_data=False,
        pipeline=test_pipeline,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator

# default_runtime
default_scope = 'mmpose'
default_hooks = dict()
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
