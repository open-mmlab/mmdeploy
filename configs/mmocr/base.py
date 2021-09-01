_base_ = ['../_base_/torch2onnx.py']
codebase = 'mmocr'

# 'TextDetection' or 'TextRecognition'
task = 'TextDetection'

pytorch2onnx = dict(
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {
        0: 'batch',
        2: 'height',
        3: 'width'
    }})

if task == 'TextRecognition':
    pytorch2onnx['dynamic_axes'] = {'input': {0: 'batch', 3: 'width'}}
