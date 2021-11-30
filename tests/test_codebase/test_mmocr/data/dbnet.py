# Copyright (c) OpenMMLab. All rights reserved.
model = dict(
    type='DBNet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        norm_eval=False,
        style='caffe'),
    neck=dict(type='FPNC', in_channels=[2, 4, 8, 16], lateral_channels=8),
    bbox_head=dict(
        type='DBHead',
        text_repr_type='quad',
        in_channels=8,
        loss=dict(type='DBLoss', alpha=5.0, beta=10.0, bbce_loss=True)),
    train_cfg=None,
    test_cfg=None)

dataset_type = 'IcdarDataset'
data_root = 'tests/test_codebase/test_mmocr/data'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(128, 64),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(256, 128), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    test_dataloader=dict(samples_per_gpu=1),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/text_detection.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=100, metric='hmean-iou')
