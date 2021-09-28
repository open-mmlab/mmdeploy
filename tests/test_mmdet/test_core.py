import importlib

import mmcv
import pytest
import torch

from mmdeploy.mmdet.core.post_processing.bbox_nms import multiclass_nms
from mmdeploy.utils.test import WrapFunction, get_rewrite_outputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('tensorrt'), reason='requires tensorrt')
def test_multiclass_nms_static():

    import tensorrt as trt
    deploy_cfg = mmcv.Config(
        dict(
            onnx_config=dict(
                output_names=['dets', 'labels'], input_shape=None),
            backend_config=dict(
                type='tensorrt',
                common_config=dict(
                    fp16_mode=False,
                    log_level=trt.Logger.INFO,
                    max_workspace_size=1 << 30),
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            boxes=dict(
                                min_shape=[1, 500, 4],
                                opt_shape=[1, 500, 4],
                                max_shape=[1, 500, 4]),
                            scores=dict(
                                min_shape=[1, 500, 80],
                                opt_shape=[1, 500, 80],
                                max_shape=[1, 500, 80])))
                ]),
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
                ))))

    boxes = torch.rand(1, 500, 4).cuda()
    scores = torch.rand(1, 500, 80).cuda()
    max_output_boxes_per_class = 200
    keep_top_k = 100
    wrapped_func = WrapFunction(
        multiclass_nms,
        max_output_boxes_per_class=max_output_boxes_per_class,
        keep_top_k=keep_top_k)
    rewrite_outputs = get_rewrite_outputs(
        wrapped_func,
        model_inputs={
            'boxes': boxes,
            'scores': scores
        },
        deploy_cfg=deploy_cfg)

    assert rewrite_outputs is not None, 'Got unexpected rewrite '\
        'outputs: {}'.format(rewrite_outputs)
