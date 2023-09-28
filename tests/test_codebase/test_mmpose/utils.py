# Copyright (c) OpenMMLab. All rights reserved.
import mmengine
import numpy
import torch
from mmengine.structures import InstanceData, PixelData

from mmdeploy.apis import build_task_processor
from mmdeploy.utils import IR, Backend, Codebase, Task, load_config


def generate_datasample(img_size, heatmap_size=(64, 48)):
    from mmpose.structures import PoseDataSample
    h, w = img_size[:2]
    metainfo = dict(
        img_shape=(h, w, 3),
        crop_size=(h, w),
        input_size=(h, w),
        input_center=numpy.asarray((h / 2, w / 2)),
        input_scale=numpy.asarray((h, w)),
        heatmap_size=heatmap_size)
    pred_instances = InstanceData()
    pred_instances.bboxes = numpy.array([[0.0, 0.0, 1.0, 1.0]])
    pred_instances.bbox_scales = torch.ones(1, 2).numpy()
    pred_instances.bbox_scores = torch.ones(1).numpy()
    pred_instances.bbox_centers = torch.ones(1, 2).numpy()
    pred_instances.keypoints = torch.rand((1, 17, 2))
    pred_instances.keypoints_visible = torch.rand((1, 17, 1))
    gt_fields = PixelData()
    gt_fields.heatmaps = torch.rand((17, 64, 48))
    data_sample = PoseDataSample(metainfo=metainfo)
    data_sample.pred_instances = pred_instances
    data_sample.gt_instances = pred_instances
    data_sample.gt_fields = gt_fields
    return data_sample


def generate_mmpose_deploy_config(backend=Backend.ONNXRUNTIME.value,
                                  cfg_options=None):
    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(type=backend),
            codebase_config=dict(
                type=Codebase.MMPOSE.value, task=Task.POSE_DETECTION.value),
            onnx_config=dict(
                type=IR.ONNX.value,
                export_params=True,
                keep_initializers_as_inputs=False,
                opset_version=11,
                input_shape=None,
                input_names=['input'],
                output_names=['output'])))

    if cfg_options is not None:
        deploy_cfg.update(cfg_options)

    return deploy_cfg


def generate_mmpose_task_processor(model_cfg=None, deploy_cfg=None):

    if model_cfg is None:
        model_cfg = 'tests/test_codebase/test_mmpose/data/model.py'
    if deploy_cfg is None:
        deploy_cfg = generate_mmpose_deploy_config()
    model_cfg, deploy_cfg = load_config(model_cfg, deploy_cfg)
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
    return task_processor
