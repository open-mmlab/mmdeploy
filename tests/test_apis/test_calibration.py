import os.path as osp
import tempfile

import mmcv
import pytest
import torch

from mmdeploy.apis import create_calib_table

try:
    from mmdeploy.apis.tensorrt.calib_utils import HDF5Calibrator
except ImportError:
    pytest.skip(
        'TensorRT should be installed from source.', allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip(
        'CUDA is required for this test module', allow_module_level=True)

calib_file = tempfile.NamedTemporaryFile(suffix='.h5').name
data_prefix = 'tests/data/tiger'
ann_file = 'tests/data/annotation.json'


def get_end2end_deploy_cfg():
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


def get_partition_deploy_cfg():
    deploy_cfg = get_end2end_deploy_cfg()
    deploy_cfg._cfg_dict['partition_config'] = dict(
        type='two_stage', apply_marks=True)
    return deploy_cfg


def get_model_cfg():
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


def test_create_calib_end2end():
    model_cfg = get_model_cfg()
    deploy_cfg = get_end2end_deploy_cfg()
    create_calib_table(
        calib_file,
        deploy_cfg,
        model_cfg,
        None,
        dataset_cfg=model_cfg,
        dataset_type='val',
        device='cuda:0')
    assert osp.exists(calib_file)

    calibrator = HDF5Calibrator(calib_file, None, 'end2end')
    assert calibrator is not None
    assert calibrator.calib_data['input']
    assert calibrator.calib_data['input']['0']


def test_create_calib_parittion():
    model_cfg = get_model_cfg()
    deploy_cfg = get_partition_deploy_cfg()
    create_calib_table(
        calib_file,
        deploy_cfg,
        model_cfg,
        None,
        dataset_cfg=model_cfg,
        dataset_type='val',
        device='cuda:0')
    assert osp.exists(calib_file)

    input_names = ['input', 'bbox_feats']
    for i in range(2):
        partition_name = f'partition{i}'
        calibrator = HDF5Calibrator(calib_file, None, partition_name)
        assert calibrator is not None
        assert calibrator.calib_data[input_names[i]]
        assert calibrator.calib_data[input_names[i]]['0']
