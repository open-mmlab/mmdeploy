_base_ = ['../../_base_/onnx_config.py']
codebase_config = dict(
    type='mmdet3d', task='VoxelDetection', model_type='end2end')
onnx_config = dict(
    input_names=['voxels', 'num_points', 'coors'],
    # need to change output_names for head with multi-level features
    output_names=['cls_score0', 'bbox_pred0', 'dir_cls_pred0'])
