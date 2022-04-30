# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import pytest
import torch

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import (WrapFunction, backend_checker, get_onnx_model,
                                 get_rewrite_outputs)

try:
    import_codebase(Codebase.MMROTATE)
except ImportError:
    pytest.skip(
        f'{Codebase.MMROTATE} is not installed.', allow_module_level=True)


@backend_checker(Backend.ONNXRUNTIME)
def test_multiclass_nms_rotated():
    from mmdeploy.codebase.mmrotate.core import multiclass_nms_rotated
    deploy_cfg = mmcv.Config(
        dict(
            onnx_config=dict(output_names=None, input_shape=None),
            backend_config=dict(
                type='onnxruntime',
                common_config=dict(
                    fp16_mode=False, max_workspace_size=1 << 20),
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            boxes=dict(
                                min_shape=[1, 5, 5],
                                opt_shape=[1, 5, 5],
                                max_shape=[1, 5, 5]),
                            scores=dict(
                                min_shape=[1, 5, 8],
                                opt_shape=[1, 5, 8],
                                max_shape=[1, 5, 8])))
                ]),
            codebase_config=dict(
                type='mmrotate',
                task='RotatedDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    pre_top_k=-1,
                    keep_top_k=10,
                ))))

    boxes = torch.rand(1, 5, 5)
    scores = torch.rand(1, 5, 8)
    keep_top_k = 10
    wrapped_func = WrapFunction(multiclass_nms_rotated, keep_top_k=keep_top_k)
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={
            'boxes': boxes,
            'scores': scores
        },
        deploy_cfg=deploy_cfg)

    assert rewrite_outputs is not None, 'Got unexpected rewrite '\
        'outputs: {}'.format(rewrite_outputs)


@backend_checker(Backend.ONNXRUNTIME)
@pytest.mark.parametrize('pre_top_k', [-1, 1000])
def test_multiclass_nms_rotated_with_keep_top_k(pre_top_k):
    backend_type = 'onnxruntime'

    from mmdeploy.codebase.mmrotate.core import multiclass_nms_rotated
    keep_top_k = 15
    deploy_cfg = mmcv.Config(
        dict(
            onnx_config=dict(
                output_names=None,
                input_shape=None,
                dynamic_axes=dict(
                    boxes={
                        0: 'batch_size',
                        1: 'num_boxes'
                    },
                    scores={
                        0: 'batch_size',
                        1: 'num_boxes',
                        2: 'num_classes'
                    },
                ),
            ),
            backend_config=dict(type=backend_type),
            codebase_config=dict(
                type='mmrotate',
                task='RotatedDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    pre_top_k=pre_top_k,
                    keep_top_k=keep_top_k,
                ))))

    num_classes = 5
    num_boxes = 2
    batch_size = 1
    export_boxes = torch.rand(batch_size, num_boxes, 5)
    export_scores = torch.ones(batch_size, num_boxes, num_classes)
    model_inputs = {'boxes': export_boxes, 'scores': export_scores}

    wrapped_func = WrapFunction(multiclass_nms_rotated, keep_top_k=keep_top_k)

    onnx_model_path = get_onnx_model(
        wrapped_func, model_inputs=model_inputs, deploy_cfg=deploy_cfg)

    num_boxes = 100
    test_boxes = torch.rand(batch_size, num_boxes, 5)
    test_scores = torch.ones(batch_size, num_boxes, num_classes)
    model_inputs = {'boxes': test_boxes, 'scores': test_scores}

    import mmdeploy.backend.onnxruntime as ort_apis
    backend_model = ort_apis.ORTWrapper(onnx_model_path, 'cuda:0', None)
    output = backend_model.forward(model_inputs)
    output = backend_model.output_to_list(output)
    dets = output[0]

    # Subtract 1 dim since we pad the tensors
    assert dets.shape[1] - 1 < keep_top_k, \
        'multiclass_nms_rotated returned more values than "keep_top_k"\n' \
        f'dets.shape: {dets.shape}\n' \
        f'keep_top_k: {keep_top_k}'
