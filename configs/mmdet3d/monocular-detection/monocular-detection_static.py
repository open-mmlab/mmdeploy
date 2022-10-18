_base_ = ['../../_base_/onnx_config.py']

onnx_config = dict(
    input_names=['img', 'cam2img', 'cam2img_inverse'],
    output_names=['bboxes', 'scores', 'labels', 'dir_scores', 'attrs'],
    input_shape=None,
)
codebase_config = dict(
    type='mmdet3d',
    task='MonocularDetection',
    model_type='end2end',
    ann_file='tests/test_codebase/test_mmdet3d/data/nuscenes/n015-2018-07-24'
    '-11-22-45+0800__CAM_BACK__1532402927637525_mono3d.coco.json',
)
