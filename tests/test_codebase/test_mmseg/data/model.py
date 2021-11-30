# Copyright (c) OpenMMLab. All rights reserved.
# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '.'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (128, 128)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(128, 128),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='',
        pipeline=test_pipeline))

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.01)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='FastSCNN',
        downsample_dw_channels=(32, 48),
        global_in_channels=64,
        global_block_channels=(64, 96, 128),
        global_block_strides=(2, 2, 1),
        global_out_channels=128,
        higher_in_channels=64,
        lower_in_channels=128,
        fusion_out_channels=128,
        out_indices=(0, 1, 2),
        norm_cfg=norm_cfg,
        align_corners=False),
    decode_head=dict(
        type='DepthwiseSeparableFCNHead',
        in_channels=128,
        channels=128,
        concat_input=False,
        num_classes=19,
        in_index=-1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
