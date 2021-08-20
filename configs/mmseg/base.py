_base_ = ['../_base_/torch2onnx.py']
codebase = 'mmseg'
pytorch2onnx = dict(
    input_names=['input'],
    output_names=['output'],
)
