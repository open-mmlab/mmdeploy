_base_ = ['../../_base_/onnx_config.py']
codebase_config = dict(
    type='mmdet3d', task='VoxelDetection', model_type='end2end')
onnx_config = dict(
    input_names=['placeholder', 'num_points', 'voxels', 'coors'],
    output_names=['cls_score', 'bbox_pred', 'dir_cls_pred'])
