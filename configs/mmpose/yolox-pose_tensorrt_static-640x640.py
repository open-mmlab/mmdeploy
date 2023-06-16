_base_ = ['./pose-detection_static.py', '../_base_/backends/tensorrt.py']

onnx_config = dict(
    input_shape=None,
    output_names=[
        'pred_bbox', 'pred_label', 'pred_score', 'pred_keypoints',
        'pred_keypoint_scores'
    ])
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 640, 640],
                    opt_shape=[1, 3, 640, 640],
                    max_shape=[1, 3, 640, 640])))
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
