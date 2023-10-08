# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile

import numpy
import onnxruntime
import pytest
import torch

from mmdeploy.backend.onnxruntime.init_plugins import get_ops_path
from mmdeploy.mmcv.ops import ONNXNMSMatchOp

cur_dir = os.path.dirname(os.path.abspath(__file__))
boxes = torch.tensor([
    [
        [291.1746, 316.2263, 343.5029, 347.7312],
        [288.4846, 315.0447, 343.7267, 346.5630],
        [288.5307, 318.1989, 341.6425, 349.7222],
        [918.9102, 83.7463, 933.3920, 164.9041],
        [895.5786, 78.2361, 907.8049, 172.0883],
        [292.5816, 316.5563, 340.3462, 352.9989],
        [609.4592, 83.5447, 631.2532, 144.0749],
        [917.7308, 85.5870, 933.2839, 168.4530],
        [895.5138, 79.3596, 908.2865, 171.0418],
        [291.4747, 318.6987, 347.1208, 349.5754],
    ],
    [
        [291.1746, 316.2263, 343.5029, 347.7312],
        [288.4846, 315.0447, 343.7267, 346.5630],
        [288.5307, 318.1989, 341.6425, 349.7222],
        [918.9102, 83.7463, 933.3920, 164.9041],
        [895.5786, 78.2361, 907.8049, 172.0883],
        [292.5816, 316.5563, 340.3462, 352.9989],
        [609.4592, 83.5447, 631.2532, 144.0749],
        [917.7308, 85.5870, 933.2839, 168.4530],
        [895.5138, 79.3596, 908.2865, 171.0418],
        [291.4747, 318.6987, 347.1208, 349.5754],
    ],
])
scores = torch.tensor([
    [
        [0.9577, 0.9745, 0.3030, 0.6589, 0.2742],
        [0.1618, 0.7963, 0.5124, 0.6964, 0.6850],
        [0.8425, 0.4843, 0.9489, 0.8068, 0.7340],
        [0.7337, 0.4340, 0.9923, 0.0704, 0.4506],
        [0.3090, 0.5606, 0.6939, 0.3764, 0.6920],
        [0.0044, 0.7986, 0.2221, 0.2782, 0.4378],
        [0.7293, 0.2735, 0.8381, 0.0264, 0.6278],
        [0.7144, 0.1066, 0.4125, 0.4041, 0.8819],
        [0.4963, 0.7891, 0.6908, 0.1499, 0.5584],
        [0.4385, 0.6035, 0.0508, 0.0662, 0.5938],
    ],
    [
        [0.9577, 0.9745, 0.3030, 0.6589, 0.2742],
        [0.1618, 0.7963, 0.5124, 0.6964, 0.6850],
        [0.8425, 0.4843, 0.9489, 0.8068, 0.7340],
        [0.7337, 0.4340, 0.9923, 0.0704, 0.4506],
        [0.3090, 0.5606, 0.6939, 0.3764, 0.6920],
        [0.0044, 0.7986, 0.2221, 0.2782, 0.4378],
        [0.7293, 0.2735, 0.8381, 0.0264, 0.6278],
        [0.7144, 0.1066, 0.4125, 0.4041, 0.8819],
        [0.4963, 0.7891, 0.6908, 0.1499, 0.5584],
        [0.4385, 0.6035, 0.0508, 0.0662, 0.5938],
    ],
])
scores = scores.permute(0, 2, 1)
iou_threshold = torch.tensor([0.1])
score_threshold = torch.tensor([0.1])
match_op = ONNXNMSMatchOp.apply


class test_ONNX_Match(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, boxes, scores, iou_threshold, score_threshold):
        return match_op(boxes, scores, iou_threshold, score_threshold)


@pytest.mark.skipif(
    reason='Need to build onnxrumtime custom op',
    condition=get_ops_path() == '')
def test_nms_match():
    print('Running compilation...')
    # here is a PyTorch test
    model = test_ONNX_Match()
    torch_output = model(boxes, scores, iou_threshold,
                         score_threshold).detach().numpy()
    # export the onnx file with a tempfile
    temp_onnx = tempfile.NamedTemporaryFile(
        suffix='.onnx', delete=False, mode='wb', dir=cur_dir)
    input_name = ['boxes', 'scores', 'iou_thr', 'score_thr']
    torch.onnx.export(
        model,
        (boxes, scores, iou_threshold, score_threshold),
        temp_onnx.name,
        input_names=input_name,
    )
    temp_onnx.close()
    options = onnxruntime.SessionOptions()
    options.register_custom_ops_library(get_ops_path())

    sess = onnxruntime.InferenceSession(
        temp_onnx.name, options, providers=['CPUExecutionProvider'])
    ort_output = sess.run(
        None,
        {
            'boxes': boxes.numpy(),
            'scores': scores.numpy(),
            'iou_thr': iou_threshold.numpy(),
            'score_thr': score_threshold.numpy(),
        },
    )

    assert numpy.array_equal(
        numpy.array(torch_output),
        numpy.array(ort_output[0])), 'list are not equal'
    os.remove(temp_onnx.name)
