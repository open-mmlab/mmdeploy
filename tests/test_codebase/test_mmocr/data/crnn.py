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

dictionary = dict(
    type='Dictionary',
    dict_file='tests/test_codebase/test_mmocr/data/lower_english_digits.txt',
    with_padding=True)

default_scope = 'mmocr'
model = dict(
    type='mmocr.CRNN',
    preprocessor=None,
    backbone=dict(type='MiniVGG', leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(
        type='CRNNDecoder',
        in_channels=512,
        rnn_flag=True,
        module_loss=dict(type='CTCModuleLoss', letter_case='lower'),
        postprocessor=dict(type='CTCPostProcessor'),
        dictionary=dictionary),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor', mean=[127], std=[127]))

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

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='grayscale',
        file_client_args=dict(backend='disk')),
    dict(
        type='RescaleToHeight',
        height=32,
        min_width=32,
        max_width=None,
        width_divisor=16),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('ori_shape', 'img_shape', 'valid_ratio'))
]

val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')
param_scheduler = [dict(type='ConstantLR', factor=1.0)]
file_client_args = dict(backend='disk')

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

visualizer = dict(type='TextRecogLocalVisualizer', name='visualizer')
