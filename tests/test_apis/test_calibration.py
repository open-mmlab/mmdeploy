# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from multiprocessing import Process

import mmcv
import pytest

from mmdeploy.apis import create_calib_input_data

calib_file = tempfile.NamedTemporaryFile(suffix='.h5').name
ann_file = 'tests/data/annotation.json'


@pytest.fixture
def deploy_cfg():
    deploy_cfg = mmcv.Config(
        dict(
            onnx_config=dict(
                dynamic_axes={
                    'input': {
                        0: 'batch',
                        2: 'height',
                        3: 'width'
                    },
                    'dets': {
                        0: 'batch',
                        1: 'num_dets',
                    },
                    'labels': {
                        0: 'batch',
                        1: 'num_dets',
                    },
                },
                type='onnx',
                export_params=True,
                keep_initializers_as_inputs=False,
                opset_version=11,
                save_file='end2end.onnx',
                input_names=['input'],
                output_names=['dets', 'labels'],
                input_shape=None),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=-1,
                    keep_top_k=100,
                    background_label_id=-1,
                )),
            backend_config=dict(type='onnxruntime')))
    return deploy_cfg


@pytest.fixture
def partition_deploy_cfg(deploy_cfg):
    deploy_cfg._cfg_dict['partition_config'] = dict(
        type='two_stage', apply_marks=True)
    return deploy_cfg


@pytest.fixture
def model_cfg():
    dataset_type = 'CustomDataset'
    data_root = 'tests/data/'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1, 1),
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
            data=dict(
                samples_per_gpu=1,
                workers_per_gpu=1,
                val=dict(
                    type=dataset_type,
                    ann_file=ann_file,
                    img_prefix=data_root,
                    pipeline=test_pipeline)),

            # model settings
            model=dict(
                type='FasterRCNN',
                backbone=dict(
                    type='ResNet',
                    depth=50,
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    frozen_stages=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    norm_eval=True,
                    style='pytorch',
                    init_cfg=dict(type='Pretrained')),
                neck=dict(
                    type='FPN',
                    in_channels=[256, 512, 1024, 2048],
                    out_channels=256,
                    num_outs=5),
                rpn_head=dict(
                    type='RPNHead',
                    in_channels=256,
                    feat_channels=256,
                    anchor_generator=dict(
                        type='AnchorGenerator',
                        scales=[8],
                        ratios=[0.5, 1.0, 2.0],
                        strides=[4, 8, 16, 32, 64]),
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[.0, .0, .0, .0],
                        target_stds=[1.0, 1.0, 1.0, 1.0]),
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=True,
                        loss_weight=1.0),
                    loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
                roi_head=dict(
                    type='StandardRoIHead',
                    bbox_roi_extractor=dict(
                        type='SingleRoIExtractor',
                        roi_layer=dict(
                            type='RoIAlign', output_size=7, sampling_ratio=0),
                        out_channels=256,
                        featmap_strides=[4, 8, 16, 32]),
                    bbox_head=dict(
                        type='Shared2FCBBoxHead',
                        in_channels=256,
                        fc_out_channels=1024,
                        roi_feat_size=7,
                        num_classes=80,
                        bbox_coder=dict(
                            type='DeltaXYWHBBoxCoder',
                            target_means=[0., 0., 0., 0.],
                            target_stds=[0.1, 0.1, 0.2, 0.2]),
                        reg_class_agnostic=False,
                        loss_cls=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=1.0),
                        loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
                # model testing settings
                test_cfg=dict(
                    rpn=dict(
                        nms_pre=1000,
                        max_per_img=1000,
                        nms=dict(type='nms', iou_threshold=0.7),
                        min_bbox_size=0),
                    rcnn=dict(
                        score_thr=0.05,
                        nms=dict(type='nms', iou_threshold=0.5),
                        max_per_img=100)))))

    return model_cfg


def run_test_create_calib_end2end(deploy_cfg, model_cfg):
    import h5py
    create_calib_input_data(
        calib_file,
        deploy_cfg,
        model_cfg,
        None,
        dataset_cfg=model_cfg,
        dataset_type='val',
        device='cpu')
    assert osp.exists(calib_file)

    with h5py.File(calib_file, mode='r') as calibrator:
        assert calibrator['calib_data'] is not None
        assert calibrator['calib_data']['end2end'] is not None
        assert calibrator['calib_data']['end2end']['input'] is not None
        assert calibrator['calib_data']['end2end']['input']['0'] is not None


# Because Faster-RCNN needs too much memory on GPU, we need to run tests in a
# new process.


def test_create_calib_end2end(deploy_cfg, model_cfg):
    p = Process(
        target=run_test_create_calib_end2end,
        kwargs=dict(deploy_cfg=deploy_cfg, model_cfg=model_cfg))
    try:
        p.start()
    finally:
        p.join()


def run_test_create_calib_parittion(partition_deploy_cfg, model_cfg):
    import h5py
    deploy_cfg = partition_deploy_cfg
    create_calib_input_data(
        calib_file,
        deploy_cfg,
        model_cfg,
        None,
        dataset_cfg=model_cfg,
        dataset_type='val',
        device='cpu')
    assert osp.exists(calib_file)

    input_names = ['input', 'bbox_feats']
    with h5py.File(calib_file, mode='r') as calibrator:
        assert calibrator['calib_data'] is not None
        calib_data = calibrator['calib_data']
        for i in range(2):
            partition_name = f'partition{i}'
            assert calib_data[partition_name] is not None
            assert calib_data[partition_name][input_names[i]] is not None
            assert calib_data[partition_name][input_names[i]]['0'] is not None


def test_create_calib_parittion(partition_deploy_cfg, model_cfg):
    p = Process(
        target=run_test_create_calib_parittion,
        kwargs=dict(
            partition_deploy_cfg=partition_deploy_cfg, model_cfg=model_cfg))
    try:
        p.start()
    finally:
        p.join()
