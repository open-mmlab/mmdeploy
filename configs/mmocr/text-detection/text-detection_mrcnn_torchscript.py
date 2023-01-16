_base_ = [
    '../../_base_/torchscript_config.py',
    '../../_base_/backends/torchscript.py'
]

ir_config = dict(input_shape=None, output_names=['dets', 'labels', 'masks'])
codebase_config = dict(
    type='mmocr',
    task='TextDetection',
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
        export_postprocess_mask=False))
