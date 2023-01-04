# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.utils import Backend
from mmdeploy.utils.test import (WrapFunction, WrapModel, backend_checker,
                                 check_backend, get_onnx_model,
                                 get_rewrite_outputs)


@backend_checker(Backend.ONNXRUNTIME)
def test_multiclass_nms_rotated():
    from mmdeploy.codebase.mmrotate.core.post_processing import \
        multiclass_nms_rotated
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

    from mmdeploy.codebase.mmrotate.core.post_processing import \
        multiclass_nms_rotated
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
    backend_model = ort_apis.ORTWrapper(onnx_model_path, 'cpu', None)
    output = backend_model.forward(model_inputs)
    output = backend_model.output_to_list(output)
    dets = output[0]

    # Subtract 1 dim since we pad the tensors
    assert dets.shape[1] - 1 < keep_top_k, \
        'multiclass_nms_rotated returned more values than "keep_top_k"\n' \
        f'dets.shape: {dets.shape}\n' \
        f'keep_top_k: {keep_top_k}'


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
@pytest.mark.parametrize('add_ctr_clamp', [True, False])
@pytest.mark.parametrize('max_shape,proj_xy,edge_swap',
                         [(None, False, False),
                          (torch.tensor([100, 200]), True, True)])
def test_delta_xywha_rbbox_coder_delta2bbox(backend_type: Backend,
                                            add_ctr_clamp: bool,
                                            max_shape: tuple, proj_xy: bool,
                                            edge_swap: bool):
    check_backend(backend_type)
    deploy_cfg = mmcv.Config(
        dict(
            onnx_config=dict(output_names=None, input_shape=None),
            backend_config=dict(type=backend_type.value, model_inputs=None),
            codebase_config=dict(type='mmrotate', task='RotatedDetection')))

    # wrap function to enable rewrite
    def delta2bbox(*args, **kwargs):
        import mmrotate
        return mmrotate.core.bbox.coder.delta_xywha_rbbox_coder.delta2bbox(
            *args, **kwargs)

    rois = torch.rand(5, 5)
    deltas = torch.rand(5, 5)
    original_outputs = delta2bbox(
        rois,
        deltas,
        max_shape=max_shape,
        add_ctr_clamp=add_ctr_clamp,
        proj_xy=proj_xy,
        edge_swap=edge_swap)

    # wrap function to nn.Module, enable torch.onnx.export
    wrapped_func = WrapFunction(
        delta2bbox,
        max_shape=max_shape,
        add_ctr_clamp=add_ctr_clamp,
        proj_xy=proj_xy,
        edge_swap=edge_swap)
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_func,
        model_inputs={
            'rois': rois.unsqueeze(0),
            'deltas': deltas.unsqueeze(0)
        },
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        model_output = original_outputs.squeeze().cpu().numpy()
        rewrite_output = rewrite_outputs[0].squeeze().cpu().numpy()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)
    else:
        assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_delta_midpointoffset_rbbox_delta2bbox(backend_type: Backend):
    check_backend(backend_type)
    deploy_cfg = mmcv.Config(
        dict(
            onnx_config=dict(output_names=None, input_shape=None),
            backend_config=dict(type=backend_type.value, model_inputs=None),
            codebase_config=dict(type='mmrotate', task='RotatedDetection')))

    # wrap function to enable rewrite
    def delta2bbox(*args, **kwargs):
        import mmrotate
        return mmrotate.core.bbox.coder.delta_midpointoffset_rbbox_coder\
            .delta2bbox(*args, **kwargs)

    rois = torch.rand(5, 4)
    deltas = torch.rand(5, 6)
    original_outputs = delta2bbox(rois, deltas, version='le90')

    # wrap function to nn.Module, enable torch.onnx.export
    wrapped_func = WrapFunction(delta2bbox, version='le90')
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_func,
        model_inputs={
            'rois': rois.unsqueeze(0),
            'deltas': deltas.unsqueeze(0)
        },
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        model_output = original_outputs.squeeze().cpu().numpy()
        rewrite_output = rewrite_outputs[0].squeeze().cpu().numpy()
        assert np.allclose(
            model_output[:, :4], rewrite_output[:, :4], rtol=1e-03, atol=1e-05)
    else:
        assert rewrite_outputs is not None


@backend_checker(Backend.ONNXRUNTIME)
def test_fake_multiclass_nms_rotated():
    from mmdeploy.codebase.mmrotate.core.post_processing import \
        fake_multiclass_nms_rotated
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
    wrapped_func = WrapFunction(
        fake_multiclass_nms_rotated, keep_top_k=keep_top_k)
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={
            'boxes': boxes,
            'scores': scores
        },
        deploy_cfg=deploy_cfg)

    assert rewrite_outputs is not None, 'Got unexpected rewrite '\
        'outputs: {}'.format(rewrite_outputs)


@pytest.mark.parametrize('backend_type', [Backend.TENSORRT])
def test_poly2obb_le90(backend_type: Backend):
    check_backend(backend_type)
    polys = torch.rand(1, 10, 8)
    deploy_cfg = mmcv.Config(
        dict(
            onnx_config=dict(output_names=None, input_shape=None),
            backend_config=dict(
                type=backend_type.value,
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            polys=dict(
                                min_shape=polys.shape,
                                opt_shape=polys.shape,
                                max_shape=polys.shape)))
                ]),
            codebase_config=dict(type='mmrotate', task='RotatedDetection')))

    # import rewriter
    from mmdeploy.codebase import Codebase, import_codebase
    import_codebase(Codebase.MMROTATE)

    # wrap function to enable rewrite
    def poly2obb_le90(*args, **kwargs):
        import mmrotate
        return mmrotate.core.bbox.transforms.poly2obb_le90(*args, **kwargs)

    # wrap function to nn.Module, enable torch.onnx.export
    wrapped_func = WrapFunction(poly2obb_le90)
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'polys': polys},
        deploy_cfg=deploy_cfg,
        run_with_backend=False)

    assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_poly2obb_le135(backend_type: Backend):
    check_backend(backend_type)
    polys = torch.rand(1, 10, 8)
    deploy_cfg = mmcv.Config(
        dict(
            onnx_config=dict(output_names=None, input_shape=None),
            backend_config=dict(
                type=backend_type.value,
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            polys=dict(
                                min_shape=polys.shape,
                                opt_shape=polys.shape,
                                max_shape=polys.shape)))
                ]),
            codebase_config=dict(type='mmrotate', task='RotatedDetection')))

    # wrap function to enable rewrite
    def poly2obb_le135(*args, **kwargs):
        import mmrotate
        return mmrotate.core.bbox.transforms.poly2obb_le135(*args, **kwargs)

    # wrap function to nn.Module, enable torch.onnx.export
    wrapped_func = WrapFunction(poly2obb_le135)
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'polys': polys},
        deploy_cfg=deploy_cfg,
        run_with_backend=False)

    assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_obb2poly_le135(backend_type: Backend):
    check_backend(backend_type)
    rboxes = torch.rand(1, 10, 5)
    deploy_cfg = mmcv.Config(
        dict(
            onnx_config=dict(output_names=None, input_shape=None),
            backend_config=dict(
                type=backend_type.value,
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            rboxes=dict(
                                min_shape=rboxes.shape,
                                opt_shape=rboxes.shape,
                                max_shape=rboxes.shape)))
                ]),
            codebase_config=dict(type='mmrotate', task='RotatedDetection')))

    # wrap function to enable rewrite
    def obb2poly_le135(*args, **kwargs):
        import mmrotate
        return mmrotate.core.bbox.transforms.obb2poly_le135(*args, **kwargs)

    # wrap function to nn.Module, enable torch.onnx.export
    wrapped_func = WrapFunction(obb2poly_le135)
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'rboxes': rboxes},
        deploy_cfg=deploy_cfg,
        run_with_backend=False)

    assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_gvfixcoder__decode(backend_type: Backend):
    check_backend(backend_type)

    deploy_cfg = mmcv.Config(
        dict(
            onnx_config=dict(output_names=['output'], input_shape=None),
            backend_config=dict(type=backend_type.value),
            codebase_config=dict(type='mmrotate', task='RotatedDetection')))

    from mmrotate.core.bbox import GVFixCoder
    coder = GVFixCoder(angle_range='le90')

    hbboxes = torch.rand(1, 10, 4)
    fix_deltas = torch.rand(1, 10, 4)

    wrapped_model = WrapModel(coder, 'decode')
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model,
        model_inputs={
            'hbboxes': hbboxes,
            'fix_deltas': fix_deltas
        },
        deploy_cfg=deploy_cfg,
        run_with_backend=False)

    assert rewrite_outputs is not None
