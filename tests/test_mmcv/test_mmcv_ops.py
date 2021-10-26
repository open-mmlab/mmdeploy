import tempfile

import onnx
import pytest
import torch

from mmdeploy.core import RewriterContext
from mmdeploy.utils.test import WrapFunction


@pytest.mark.parametrize(
    'iou_threshold, score_threshold,max_output_boxes_per_class',
    [(0.6, 0.2, 3)])
def test_ONNXNMSop(iou_threshold, score_threshold, max_output_boxes_per_class):
    boxes = torch.tensor([[[291.1746, 316.2263, 343.5029, 347.7312],
                           [288.4846, 315.0447, 343.7267, 346.5630],
                           [288.5307, 318.1989, 341.6425, 349.7222],
                           [918.9102, 83.7463, 933.3920, 164.9041],
                           [895.5786, 78.2361, 907.8049, 172.0883],
                           [292.5816, 316.5563, 340.3462, 352.9989],
                           [609.4592, 83.5447, 631.2532, 144.0749],
                           [917.7308, 85.5870, 933.2839, 168.4530],
                           [895.5138, 79.3596, 908.2865, 171.0418],
                           [291.4747, 318.6987, 347.1208, 349.5754]]])
    scores = torch.rand(1, 5, 10)

    from mmdeploy.mmcv.ops import ONNXNMSop

    def wrapped_function(torch_bboxes, torch_scores):
        return ONNXNMSop.apply(torch_bboxes, torch_scores,
                               max_output_boxes_per_class, iou_threshold,
                               score_threshold)

    wrapped_model = WrapFunction(wrapped_function).eval()
    result = wrapped_model(boxes, scores)
    assert result is not None
    onnx_file_path = tempfile.NamedTemporaryFile().name
    with RewriterContext({}, opset=11), torch.no_grad():
        torch.onnx.export(
            wrapped_model, (boxes, scores),
            onnx_file_path,
            export_params=True,
            keep_initializers_as_inputs=True,
            input_names=['boxes', 'scores'],
            output_names=['result'],
            opset_version=11)
    model = onnx.load(onnx_file_path)
    assert model.graph.node[3].op_type == 'NonMaxSuppression'
