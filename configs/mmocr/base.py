_base_ = ['../_base_/torch2onnx.py']
codebase = 'mmocr'

# 'det' for text detection and 'recog' for text recognition
algorithm_type = 'det'
pytorch2onnx = dict(
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {
        0: 'batch',
        2: 'height',
        3: 'width'
    }})

if algorithm_type == 'recog':
    pytorch2onnx['dynamic_axes'] = {'input': {0: 'batch', 3: 'width'}}
