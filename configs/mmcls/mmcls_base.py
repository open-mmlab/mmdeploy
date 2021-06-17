_base_ = ['../_base_/torch2onnx.py']
codebase = 'mmcls'
pytorch2onnx = dict(
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {
            0: 'batch'
        },
        'output': {
           0: 'batch'
        }
    }
)
