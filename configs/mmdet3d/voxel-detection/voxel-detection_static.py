_base_ = ['../../_base_/onnx_config.py']
codebase_config = dict(
    type='mmdet3d', task='VoxelDetection', model_type='end2end')
onnx_config = dict(
    input_names=['voxels', 'num_points', 'coors'],
    output_names=['bboxes', 'scores', 'labels'])
