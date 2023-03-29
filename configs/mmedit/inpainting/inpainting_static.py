_base_ = ['../../_base_/onnx_config.py']

codebase_config = dict(type='mmedit', task='Inpainting')
onnx_config = dict(
    input_names=['masked_img', 'mask'], output_names=['fake_img'])
