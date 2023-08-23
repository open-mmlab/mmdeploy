_base_ = ['../../_base_/onnx_config.py']

codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    model_type='panoptic_end2end',
    post_processing=dict(
        export_postprocess_mask=False,
        score_threshold=0.0,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
    ))
