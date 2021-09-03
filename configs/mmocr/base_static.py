_base_ = ['../_base_/torch2onnx.py']
codebase = 'mmocr'

# 'TextDetection' or 'TextRecognition'
task = 'TextDetection'
pytorch2onnx = dict(input_names=['input'], output_names=['output'])
