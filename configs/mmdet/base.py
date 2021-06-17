_base_ = ['../_base_/torch2onnx.py']
codebase = 'mmdet'
pytorch2onnx = dict(
    input_names=['input'],
    output_names=['dets', 'labels'],
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'dets': {
            0: 'batch',
            1: 'num_dets',
        },
        'labels': {
            0: 'batch',
            1: 'num_dets',
        },
    },
)
