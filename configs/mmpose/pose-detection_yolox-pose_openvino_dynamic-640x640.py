_base_ = ['./pose-detection_static.py', '../_base_/backends/openvino.py']

onnx_config = dict(
    output_names=['dets', 'keypoints'],
    dynamic_axes={
        'input': {
            0: 'batch',
        },
        'dets': {
            0: 'batch',
        },
        'keypoints': {
            0: 'batch'
        }
    })
backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 640, 640]))])

codebase_config = dict(
    post_processing=dict(
        score_threshold=0.05,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
    ))
