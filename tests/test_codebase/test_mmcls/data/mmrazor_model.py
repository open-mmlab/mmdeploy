# Copyright (c) OpenMMLab. All rights reserved.
norm_cfg = dict(type='BN')
model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='SearchableShuffleNetV2', widen_factor=1.0, norm_cfg=norm_cfg),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5),
    ),
)

mutator = dict(
    type='OneShotMutator',
    placeholder_mapping=dict(
        all_blocks=dict(
            type='OneShotOP',
            choices=dict(
                shuffle_3x3=dict(
                    type='ShuffleBlock', kernel_size=3, norm_cfg=norm_cfg),
                shuffle_5x5=dict(
                    type='ShuffleBlock', kernel_size=5, norm_cfg=norm_cfg),
                shuffle_7x7=dict(
                    type='ShuffleBlock', kernel_size=7, norm_cfg=norm_cfg),
                shuffle_xception=dict(
                    type='ShuffleXception', norm_cfg=norm_cfg),
            ))))

algorithm = dict(
    type='SPOS',
    architecture=dict(
        type='MMClsArchitecture',
        model=model,
    ),
    mutator=mutator,
    distiller=None,
    mutable_cfg='mmrazor_mutable_cfg.yaml',
    retraining=True)

evaluation = dict(interval=1000, metric='accuracy')

find_unused_parameters = True

# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        data_prefix='tests/test_codebase/test_mmcls/data/imgs/dataset',
        ann_file='tests/test_codebase/test_mmcls/data/imgs/ann.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')
