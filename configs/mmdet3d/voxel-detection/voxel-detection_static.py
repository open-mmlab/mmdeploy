_base_ = ['../../_base_/onnx_config.py']
codebase_config = dict(
    type='mmdet3d', task='VoxelDetection', model_type='end2end')
onnx_config = dict(
    input_names=['voxels', 'num_points', 'coors'],
    output_names=[
        'cls_score0',
        'bbox_pred0',
        'dir_cls_pred0',
        'cls_score1',
        'bbox_pred1',
        'dir_cls_pred1',
        'cls_score2',
        'bbox_pred2',
        'dir_cls_pred2',
    ])
