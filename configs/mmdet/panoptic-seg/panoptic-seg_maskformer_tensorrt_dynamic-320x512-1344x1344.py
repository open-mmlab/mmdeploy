_base_ = ['./panoptic-seg_maskformer_tensorrt_static-800x1344.py']
onnx_config = dict(
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'cls_logits': {
            0: 'batch',
        },
        'mask_logits': {
            0: 'batch',
            2: 'h',
            3: 'w',
        },
    },
    input_shape=None)

backend_config = dict(model_inputs=[
    dict(
        input_shapes=dict(
            input=dict(
                min_shape=[1, 3, 320, 512],
                opt_shape=[1, 3, 800, 1344],
                max_shape=[1, 3, 1344, 1344])))
])
