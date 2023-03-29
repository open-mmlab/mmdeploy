# Copyright (c) OpenMMLab. All rights reserved.
exp_name = 'deepfillv1_256x256_8x2_places'

model = dict(
    type='DeepFillv1Inpaintor',
    encdec=dict(
        type='DeepFillEncoderDecoder',
        stage1=dict(
            type='GLEncoderDecoder',
            encoder=dict(type='DeepFillEncoder', padding_mode='reflect'),
            decoder=dict(
                type='DeepFillDecoder',
                in_channels=128,
                padding_mode='reflect'),
            dilation_neck=dict(
                type='GLDilationNeck',
                in_channels=128,
                act_cfg=dict(type='ELU'),
                padding_mode='reflect')),
        stage2=dict(
            type='DeepFillRefiner',
            encoder_attention=dict(
                type='DeepFillEncoder',
                encoder_type='stage2_attention',
                padding_mode='reflect'),
            encoder_conv=dict(
                type='DeepFillEncoder',
                encoder_type='stage2_conv',
                padding_mode='reflect'),
            dilation_neck=dict(
                type='GLDilationNeck',
                in_channels=128,
                act_cfg=dict(type='ELU'),
                padding_mode='reflect'),
            contextual_attention=dict(
                type='ContextualAttentionNeck',
                in_channels=128,
                padding_mode='reflect'),
            decoder=dict(
                type='DeepFillDecoder',
                in_channels=256,
                padding_mode='reflect'))),
    disc=dict(
        type='DeepFillv1Discriminators',
        global_disc_cfg=dict(
            type='MultiLayerDiscriminator',
            in_channels=3,
            max_channels=256,
            fc_in_channels=65536,
            fc_out_channels=1,
            num_convs=4,
            norm_cfg=None,
            act_cfg=dict(type='ELU'),
            out_act_cfg=dict(type='LeakyReLU', negative_slope=0.2)),
        local_disc_cfg=dict(
            type='MultiLayerDiscriminator',
            in_channels=3,
            max_channels=512,
            fc_in_channels=32768,
            fc_out_channels=1,
            num_convs=4,
            norm_cfg=None,
            act_cfg=dict(type='ELU'),
            out_act_cfg=dict(type='LeakyReLU', negative_slope=0.2))),
    stage1_loss_type=('loss_l1_hole', 'loss_l1_valid'),
    stage2_loss_type=('loss_l1_hole', 'loss_l1_valid', 'loss_gan'),
    loss_gan=dict(type='GANLoss', gan_type='wgan', loss_weight=0.0001),
    loss_l1_hole=dict(type='L1Loss', loss_weight=1.0),
    loss_l1_valid=dict(type='L1Loss', loss_weight=1.0),
    loss_gp=dict(type='GradientPenaltyLoss', loss_weight=10.0),
    loss_disc_shift=dict(type='DiscShiftLoss', loss_weight=0.001),
    pretrained=None)

test_cfg = dict(metrics=['l1', 'psnr', 'ssim'])

test_pipeline = [
    dict(type='LoadImageFromFile', key='gt_img'),
    dict(
        type='LoadMask',
        mask_mode='bbox',
        mask_config=dict(
            max_bbox_shape=(128, 128),
            max_bbox_delta=40,
            min_margin=20,
            img_shape=(256, 256))),
    dict(type='Crop', keys=['gt_img'], crop_size=(384, 384), random_crop=True),
    dict(type='Resize', keys=['gt_img'], scale=(256, 256), keep_ratio=False),
    dict(
        type='Normalize',
        keys=['gt_img'],
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=False),
    dict(type='GetMaskedImage'),
    dict(
        type='Collect',
        keys=['gt_img', 'masked_img', 'mask', 'mask_bbox'],
        meta_keys=['gt_img_path']),
    dict(type='ImageToTensor', keys=['gt_img', 'masked_img', 'mask']),
    dict(type='ToTensor', keys=['mask_bbox'])
]
data = dict(
    test_dataloader=dict(samples_per_gpu=1),
    test=dict(
        type='ImgInpaintingDataset',
        ann_file='tests/test_codebase/test_mmedit/data/ann_file.txt',
        data_prefix='tests/test_codebase/test_mmedit/data',
        pipeline=test_pipeline,
        test_mode=True))
