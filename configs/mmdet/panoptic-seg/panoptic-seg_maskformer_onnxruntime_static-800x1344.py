_base_ = [
    '../_base_/base_panoptic-seg_static.py',
    '../../_base_/backends/onnxruntime.py'
]
onnx_config = dict(
    opset_version=13,
    output_names=['cls_logits', 'mask_logits'],
    input_shape=[1344, 800])
