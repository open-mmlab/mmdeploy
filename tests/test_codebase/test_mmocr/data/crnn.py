# Copyright (c) OpenMMLab. All rights reserved.
_base_ = []
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook')

    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# model
label_convertor = dict(
    type='CTCConvertor', dict_type='DICT36', with_unknown=False, lower=True)

model = dict(
    type='CRNNNet',
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
    loss=dict(type='CTCLoss'),
    label_convertor=label_convertor,
    pretrained=None)

train_cfg = None
test_cfg = None

# optimizer
optimizer = dict(type='Adadelta', lr=1.0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[])
total_epochs = 5

# data
img_norm_cfg = dict(mean=[127], std=[127])

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=100,
        max_width=100,
        keep_aspect_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'resize_shape', 'text', 'valid_ratio']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=32,
        max_width=None,
        keep_aspect_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'resize_shape', 'valid_ratio']),
]

dataset_type = 'OCRDataset'

test_prefix = 'tests/test_codebase/test_mmocr/data/'

test_img_prefix1 = test_prefix

test_ann_file1 = test_prefix + 'text_recognition.txt'

test1 = dict(
    type=dataset_type,
    img_prefix=test_img_prefix1,
    ann_file=test_ann_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    val=dict(
        type='UniformConcatDataset', datasets=[test1], pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset', datasets=[test1], pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')

cudnn_benchmark = True
