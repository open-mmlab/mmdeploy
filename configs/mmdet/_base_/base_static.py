_base_ = ['../../_base_/onnx_config.py']

onnx_config = dict(output_names=['dets', 'labels'], input_shape=None)
codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,  # for YOLOv3
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=-1,
        keep_top_k=100,
        background_label_id=-1,
    ))
