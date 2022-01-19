_base_ = ['../../_base_/onnx_config.py']
codebase_config = dict(type='mmdet3d', task='VoxelDetection', model_type='end2end')
onnx_config = dict(input_names=['voxels', 'point_nums', 'coors'], output_names=['bbox', 'score', 'dir_score'])
