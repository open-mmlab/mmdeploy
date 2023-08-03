_base_ = [
    '../_base_/base_panoptic-seg_static.py',
    '../../_base_/backends/tensorrt.py'
]
onnx_config = dict(
    opset_version=13,
    output_names=['cls_logits', 'mask_logits'],
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'cls_logits': {
            0: 'batch',
            1: 'query',
        },
        'mask_logits': {
            0: 'batch',
            1: 'query',
            2: 'h',
            3: 'w',
        },
    },
    input_shape=None)

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 320, 512],
                    opt_shape=[1, 3, 800, 1344],
                    max_shape=[1, 3, 1344, 1344])))
    ])
