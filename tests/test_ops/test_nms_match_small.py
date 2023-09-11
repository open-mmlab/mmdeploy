# Copyright (c) OpenMMLab. All rights reserved.
import onnxruntime
import torch

from mmdeploy.mmcv.ops import ONNXNMSMatchOp

boxes = torch.tensor([[[291.1746, 316.2263, 343.5029, 347.7312],
                       [288.4846, 315.0447, 343.7267, 346.5630],
                       [288.5307, 318.1989, 341.6425, 349.7222],
                       [918.9102, 83.7463, 933.3920, 164.9041],
                       [895.5786, 78.2361, 907.8049, 172.0883]]])
scores = torch.tensor([[[0.9577, 0.9745, 0.3030, 0.6589, 0.2742],
                        [0.1618, 0.7963, 0.5124, 0.6964, 0.6850],
                        [0.8425, 0.4843, 0.9489, 0.8068, 0.7340],
                        [0.7337, 0.4340, 0.9923, 0.0704, 0.4506],
                        [0.3090, 0.5606, 0.6939, 0.3764, 0.6920]]])
scores = scores.permute(0, 2, 1)
iou_threshold = torch.tensor([0.1])
score_threshold = torch.tensor([0.1])
match_op = ONNXNMSMatchOp.apply


class test_ONNX_Match(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, boxes, scores, iou_threshold, score_threshold):
        return match_op(boxes, scores, iou_threshold, score_threshold)


model = test_ONNX_Match()
torch.onnx.export(model, (boxes, scores, iou_threshold, score_threshold),
                  'test_onnx_match.onnx')
torch_output = model(boxes, scores, iou_threshold,
                     score_threshold).detach().numpy()

options = onnxruntime.SessionOptions()
options.register_custom_ops_library(
    'F:\\yyfanforstudy\\mmdeploy\\mmdeploy\\lib\\mmdeploy_onnxruntime_ops.dll')

sess = onnxruntime.InferenceSession(
    'test_onnx_match.onnx', options, providers=['CPUExecutionProvider'])
ort_output = sess.run(
    None, {
        'boxes': boxes.numpy(),
        'scores': scores.numpy(),
        'mmdeploy::NMSMatch_2': iou_threshold.numpy(),
        'mmdeploy::NMSMatch_3': score_threshold.numpy()
    })
print(torch_output)
print(ort_output)
print(ort_output == torch_output)
