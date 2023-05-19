# Copyright (c) OpenMMLab. All rights reserved.
default_scope = 'mmagic'
save_dir = './work_dirs'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        out_dir=save_dir,
        by_epoch=False,
        max_keep_ckpts=10,
        save_best='PSNR',
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=False)

load_from = None
resume = False
experiment_name = 'srcnn_x4k915_1xb16-1000k_div2k'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale = 4
# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='SRCNNNet',
        channels=(3, 64, 32, 3),
        kernel_sizes=(9, 1, 5),
        upscale_factor=scale),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(metrics=['PSNR'], crop_border=scale),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=128),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='PackInputs')
]
val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='PackInputs')
]

# dataset settings
dataset_type = 'BasicImageDataset'
data_root = 'data'

train_dataloader = dict(
    num_workers=4,
    batch_size=16,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='meta_info_DIV2K800sub_GT.txt',
        metainfo=dict(dataset_type='div2k', task_name='sisr'),
        data_root=data_root + '/DIV2K',
        data_prefix=dict(
            img='DIV2K_train_LR_bicubic/X4_sub', gt='DIV2K_train_HR_sub'),
        filename_tmpl=dict(img='{}', gt='{}'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='set5', task_name='sisr'),
        data_root=data_root + '/Set5',
        data_prefix=dict(img='LRbicx4', gt='GTmod12'),
        pipeline=val_pipeline))

val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR', crop_border=scale),
    dict(type='SSIM', crop_border=scale),
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=1000000, val_interval=5000)
val_cfg = dict(type='ValLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.99)))

# learning policy
param_scheduler = dict(
    type='CosineRestartLR',
    by_epoch=False,
    periods=[250000, 250000, 250000, 250000],
    restart_weights=[1, 1, 1, 1],
    eta_min=1e-7)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='PackInputs')
]

# test config for Set5
set5_data_root = 'data/Set5'
set5_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='set5', task_name='sisr'),
        data_root=set5_data_root,
        data_prefix=dict(img='imgs', gt='imgs'),
        pipeline=test_pipeline))
set5_evaluator = [
    dict(type='PSNR', crop_border=4, prefix='Set5'),
    dict(type='SSIM', crop_border=4, prefix='Set5'),
]

set14_data_root = 'data/Set14'
set14_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='set14', task_name='sisr'),
        data_root=set5_data_root,
        data_prefix=dict(img='imgs', gt='imgs'),
        pipeline=test_pipeline))
set14_evaluator = [
    dict(type='PSNR', crop_border=4, prefix='Set14'),
    dict(type='SSIM', crop_border=4, prefix='Set14'),
]

ut_data_root = 'tests/test_codebase/test_mmagic/data'
ut_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='set14', task_name='sisr'),
        data_root=ut_data_root,
        data_prefix=dict(img='imgs', gt='imgs'),
        pipeline=test_pipeline))

# test config
test_cfg = dict(type='MultiTestLoop')
test_dataloader = [ut_dataloader, ut_dataloader]
test_evaluator = [set5_evaluator, set14_evaluator]
