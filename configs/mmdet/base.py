_base_ = ['../_base_/torch2onnx.py']
codebase = 'mmdet'
pytorch2onnx = dict(
    input_names=['input'],
    output_names=['dets', 'labels'],
    dynamic_axes={'input': {
        0: 'batch',
        2: 'height',
        3: 'width'
    }},
)
post_processing = dict(
    score_threshold=0.05,
    iou_threshold=0.5,
    max_output_boxes_per_class=200,
    pre_top_k=-1,
    keep_top_k=100,
    background_label_id=-1,
)
