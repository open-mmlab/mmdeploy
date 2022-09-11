_base_ = ['../../_base_/onnx_config.py']
codebase_config = dict(
    type='mmdet3d', task='PointPillars', model_type='end2end')
onnx_config = dict(
    input_names=['points', 'voxels'],
    output_names=['scores', 'bbox_preds', 'dir_scores'])
