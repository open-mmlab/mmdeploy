_base_ = ['./mono-detection_static.py']

onnx_config = dict(
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'cls_score': {
            0: 'batch',
        },
        'bbox_pred': {
            0: 'batch',
        },
    },
    input_shape=None)
