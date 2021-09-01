import os.path as osp
import shutil

import mmcv
import numpy as np
import pytest

from mmdeploy.apis import torch2onnx

backend = 'default'
work_dir = './tmp/'
save_file = 'tmp.onnx'


@pytest.fixture(autouse=True)
def clear_workdir_after_test():
    # clear work_dir before test
    if osp.exists(work_dir):
        shutil.rmtree(work_dir)

    yield

    # clear work_dir after test
    if osp.exists(work_dir):
        shutil.rmtree(work_dir)


def test_torch2onnx_mmcls():
    codebase = 'mmcls'
    # skip if codebase is not installed
    pytest.importorskip(codebase, reason='Can not import {}.'.format(codebase))

    # deploy config
    deploy_cfg = mmcv.Config(
        dict(
            codebase=codebase,
            backend=backend,
            pytorch2onnx=dict(
                export_params=True,
                keep_initializers_as_inputs=False,
                opset_version=11,
                save_file=save_file,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {
                    0: 'batch'
                }})))

    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    dataset_type = 'ImageNet'
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', size=(256, -1)),
        dict(type='CenterCrop', crop_size=224),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])
    ]

    # model config
    model_cfg = mmcv.Config(
        dict(
            model=dict(
                type='ImageClassifier',
                backbone=dict(
                    type='ResNet',
                    depth=18,
                    num_stages=4,
                    out_indices=(3, ),
                    style='pytorch'),
                neck=dict(type='GlobalAveragePooling'),
                head=dict(
                    type='LinearClsHead',
                    num_classes=1000,
                    in_channels=512,
                    loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                    topk=(1, 5),
                )),
            dataset_type=dataset_type,
            img_norm_cfg=img_norm_cfg,
            test_pipeline=test_pipeline,
            data=dict(
                samples_per_gpu=32,
                workers_per_gpu=2,
                test=dict(
                    type=dataset_type,
                    data_prefix='data/imagenet/val',
                    ann_file='data/imagenet/meta/val.txt',
                    pipeline=test_pipeline))))

    # dummy input
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    # export
    torch2onnx(
        img,
        work_dir=work_dir,
        save_file=save_file,
        deploy_cfg=deploy_cfg,
        model_cfg=model_cfg,
        device='cpu')

    assert osp.exists(work_dir)
    assert osp.exists(osp.join(work_dir, save_file))


def test_torch2onnx_mmdet():
    codebase = 'mmdet'
    # skip if codebase is not installed
    pytest.importorskip(codebase, reason='Can not import {}.'.format(codebase))

    deploy_cfg = mmcv.Config(
        dict(
            codebase=codebase,
            backend=backend,
            pytorch2onnx=dict(
                export_params=True,
                keep_initializers_as_inputs=False,
                opset_version=11,
                save_file=save_file,
                input_names=['input'],
                output_names=['dets', 'labels'],
                dynamic_axes={'input': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'
                }}),
            post_processing=dict(
                score_threshold=0.05,
                iou_threshold=0.5,
                max_output_boxes_per_class=200,
                pre_top_k=-1,
                keep_top_k=100,
                background_label_id=-1)))

    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    dataset_type = 'CocoDataset'
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1333, 800),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    model_cfg = mmcv.Config(
        dict(
            model=dict(
                type='RetinaNet',
                pretrained='torchvision://resnet50',
                backbone=dict(
                    type='ResNet',
                    depth=50,
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    frozen_stages=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    norm_eval=True,
                    style='pytorch'),
                neck=dict(
                    type='FPN',
                    in_channels=[256, 512, 1024, 2048],
                    out_channels=256,
                    start_level=1,
                    add_extra_convs='on_input',
                    num_outs=5),
                bbox_head=dict(
                    type='RetinaHead',
                    num_classes=80,
                    in_channels=256,
                    stacked_convs=4,
                    feat_channels=256,
                    anchor_generator=dict(
                        type='AnchorGenerator',
                        octave_base_scale=4,
                        scales_per_octave=3,
                        ratios=[0.5, 1.0, 2.0],
                        strides=[8, 16, 32, 64, 128]),
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[.0, .0, .0, .0],
                        target_stds=[1.0, 1.0, 1.0, 1.0]),
                    loss_cls=dict(
                        type='FocalLoss',
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.25,
                        loss_weight=1.0),
                    loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
                # model training and testing settings
                train_cfg=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.4,
                        min_pos_iou=0,
                        ignore_iof_thr=-1),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False),
                test_cfg=dict(
                    nms_pre=1000,
                    min_bbox_size=0,
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100)),
            dataset_type=dataset_type,
            img_norm_cfg=img_norm_cfg,
            test_pipeline=test_pipeline,
            data=dict(
                samples_per_gpu=2,
                workers_per_gpu=2,
                test=dict(type=dataset_type, pipeline=test_pipeline))))

    img = np.random.randint(0, 256, (640, 960, 3), dtype=np.uint8)

    torch2onnx(
        img,
        work_dir=work_dir,
        save_file=save_file,
        deploy_cfg=deploy_cfg,
        model_cfg=model_cfg,
        device='cpu')

    assert osp.exists(work_dir)
    assert osp.exists(osp.join(work_dir, save_file))
