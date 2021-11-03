import importlib

import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.utils.test import (WrapFunction, get_onnx_model,
                                 get_rewrite_outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('tensorrt'), reason='requires tensorrt')
def test_multiclass_nms_static():

    import tensorrt as trt
    from mmdeploy.mmdet.core import multiclass_nms
    deploy_cfg = mmcv.Config(
        dict(
            onnx_config=dict(output_names=None, input_shape=None),
            backend_config=dict(
                type='tensorrt',
                common_config=dict(
                    fp16_mode=False,
                    log_level=trt.Logger.INFO,
                    max_workspace_size=1 << 20),
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            boxes=dict(
                                min_shape=[1, 5, 4],
                                opt_shape=[1, 5, 4],
                                max_shape=[1, 5, 4]),
                            scores=dict(
                                min_shape=[1, 5, 8],
                                opt_shape=[1, 5, 8],
                                max_shape=[1, 5, 8])))
                ]),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=20,
                    pre_top_k=-1,
                    keep_top_k=10,
                    background_label_id=-1,
                ))))

    boxes = torch.rand(1, 5, 4).cuda()
    scores = torch.rand(1, 5, 8).cuda()
    max_output_boxes_per_class = 20
    keep_top_k = 10
    wrapped_func = WrapFunction(
        multiclass_nms,
        max_output_boxes_per_class=max_output_boxes_per_class,
        keep_top_k=keep_top_k)
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={
            'boxes': boxes,
            'scores': scores
        },
        deploy_cfg=deploy_cfg)

    assert rewrite_outputs is not None, 'Got unexpected rewrite '\
        'outputs: {}'.format(rewrite_outputs)


@pytest.mark.parametrize('backend_type', ['onnxruntime', 'ncnn'])
def test_delta2bbox(backend_type):
    pytest.importorskip(backend_type, reason=f'requires {backend_type}')
    deploy_cfg = mmcv.Config(
        dict(
            onnx_config=dict(output_names=None, input_shape=None),
            backend_config=dict(type=backend_type, model_inputs=None),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))

    # wrap function to enable rewrite
    def delta2bbox(*args, **kwargs):
        import mmdet
        return mmdet.core.bbox.coder.delta_xywh_bbox_coder.delta2bbox(
            *args, **kwargs)

    rois = torch.rand(1, 5, 4)
    deltas = torch.rand(1, 5, 4)
    original_outputs = delta2bbox(rois, deltas)

    # wrap function to nn.Module, enable torch.onnx.export
    wrapped_func = WrapFunction(delta2bbox)
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_func,
        model_inputs={
            'rois': rois,
            'deltas': deltas
        },
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        model_output = original_outputs.squeeze().cpu().numpy()
        rewrite_output = rewrite_outputs[0].squeeze()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)
    else:
        assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type', ['onnxruntime', 'ncnn'])
def test_tblr2bbox(backend_type):
    pytest.importorskip(backend_type, reason=f'requires {backend_type}')
    deploy_cfg = mmcv.Config(
        dict(
            onnx_config=dict(output_names=None, input_shape=None),
            backend_config=dict(type=backend_type, model_inputs=None),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))

    # wrap function to enable rewrite
    def tblr2bboxes(*args, **kwargs):
        import mmdet
        return mmdet.core.bbox.coder.tblr_bbox_coder.tblr2bboxes(
            *args, **kwargs)

    priors = torch.rand(1, 5, 4)
    tblr = torch.rand(1, 5, 4)
    original_outputs = tblr2bboxes(priors, tblr)

    # wrap function to nn.Module, enable torch.onnx.export
    wrapped_func = WrapFunction(tblr2bboxes)
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_func,
        model_inputs={
            'priors': priors,
            'tblr': tblr
        },
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        model_output = original_outputs.squeeze().cpu().numpy()
        rewrite_output = rewrite_outputs[0].squeeze()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)
    else:
        assert rewrite_outputs is not None


def test_distance2bbox():
    from mmdeploy.mmdet.core import distance2bbox
    points = torch.rand(3, 2)
    distance = torch.rand(3, 4)
    bbox = distance2bbox(points, distance)
    assert bbox.shape == torch.Size([3, 4])


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_multiclass_nms_with_keep_top_k():
    backend_type = 'onnxruntime'

    from mmdeploy.mmdet.core import multiclass_nms
    max_output_boxes_per_class = 20
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
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=max_output_boxes_per_class,
                    pre_top_k=-1,
                    keep_top_k=keep_top_k,
                    background_label_id=-1,
                ))))

    num_classes = 5
    num_boxes = 2
    batch_size = 1
    export_boxes = torch.rand(batch_size, num_boxes, 4)
    export_scores = torch.ones(batch_size, num_boxes, num_classes)
    model_inputs = {'boxes': export_boxes, 'scores': export_scores}

    wrapped_func = WrapFunction(
        multiclass_nms,
        max_output_boxes_per_class=max_output_boxes_per_class,
        keep_top_k=keep_top_k)

    onnx_model_path = get_onnx_model(
        wrapped_func, model_inputs=model_inputs, deploy_cfg=deploy_cfg)

    num_boxes = 100
    test_boxes = torch.rand(batch_size, num_boxes, 4)
    test_scores = torch.ones(batch_size, num_boxes, num_classes)
    model_inputs = {'boxes': test_boxes, 'scores': test_scores}

    import mmdeploy.apis.onnxruntime as ort_apis
    backend_model = ort_apis.ORTWrapper(onnx_model_path, 0, None)
    dets, _ = backend_model.forward(model_inputs)

    assert dets.shape[1] < keep_top_k, \
        'multiclass_nms returned more values than "keep_top_k"\n' \
        f'dets.shape: {dets.shape}\n' \
        f'keep_top_k: {keep_top_k}'
