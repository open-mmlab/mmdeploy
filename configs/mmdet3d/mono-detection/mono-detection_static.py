_base_ = ['../../_base_/onnx_config.py']
codebase_config = dict(
    type='mmdet3d', task='MonoDetection', model_type='end2end')
onnx_config = dict(
    input_names=['input'], output_names=['cls_score', 'bbox_pred'])
