_base_ = ['../../_base_/onnx_config.py']
codebase_config = dict(
    type='mmdet3d', task='VoxelDetection', model_type='end2end')
onnx_config = dict(
    input_names=['identify', 'num_points', 'voxels', 'coors'],
    output_names=['cls_score', 'bbox_pred', 'dir_cls_pred'])

#(Pdb) p dir_cls_pred.shape
#torch.Size([4, 248, 216])
#(Pdb) p bbox_pred.shape
#torch.Size([14, 248, 216])
#(Pdb) p cls_score.shape
#torch.Size([2, 248, 216])
