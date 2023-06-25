# Copyright (c) OpenMMLab. All rights reserved.
import mmengine
import torch
from mmengine.structures import PixelData

from mmdeploy.apis import build_task_processor
from mmdeploy.utils import load_config


def generate_datasample(h, w):
    from mmseg.structures import SegDataSample
    metainfo = dict(img_shape=(h, w), ori_shape=(h, w), pad_shape=(h, w))
    data_sample = SegDataSample()
    data_sample.set_metainfo(metainfo)
    seg_pred = torch.randint(0, 2, (1, h, w))
    seg_gt = torch.randint(0, 2, (1, h, w))
    data_sample.set_data(dict(pred_sem_seg=PixelData(**dict(data=seg_pred))))
    data_sample.set_data(
        dict(gt_sem_seg=PixelData(**dict(data=seg_gt, metainfo=metainfo))))
    return data_sample


def generate_mmseg_deploy_config(backend='onnxruntime'):
    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(type=backend),
            codebase_config=dict(
                type='mmseg', task='Segmentation', with_argmax=False),
            onnx_config=dict(
                type='onnx',
                export_params=True,
                keep_initializers_as_inputs=False,
                opset_version=11,
                input_shape=None,
                input_names=['inputs'],
                output_names=['output'])))
    return deploy_cfg


def generate_mmseg_task_processor(model_cfg=None, deploy_cfg=None):
    if model_cfg is None:
        model_cfg = 'tests/test_codebase/test_mmseg/data/model.py'
    if deploy_cfg is None:
        deploy_cfg = generate_mmseg_deploy_config()
    model_cfg, deploy_cfg = load_config(model_cfg, deploy_cfg)
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
    return task_processor
