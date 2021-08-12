_base_ = ['../_base_/torch2onnx.py']
codebase = 'mmseg'
pytorch2onnx = dict(
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'output': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
    },
)
