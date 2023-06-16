_base_ = ['./pose-detection_static.py', '../_base_/backends/onnxruntime.py']

onnx_config = dict(
    input_shape=None,
    output_names=[
        'pred_bbox', 'pred_label', 'pred_score', 'pred_keypoints',
        'pred_keypoint_scores'
    ])

codebase_config = dict(
    type='mmpose',
    task='PoseDetection',
    model_type='yolox-pose_end2end',
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,  # for YOLOv3
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
    ))
