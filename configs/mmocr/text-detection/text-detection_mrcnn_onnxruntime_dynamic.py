_base_ = ['./text-detection_static.py', '../../_base_/backends/onnxruntime.py']
onnx_config = dict(
    output_names=['dets', 'labels', 'masks'],
    dynamic_axes=dict(
        input=dict({
            0: 'batch',
            2: 'height',
            3: 'width'
        }),
        dets=dict({
            0: 'batch',
            1: 'num_dets'
        }),
        labels=dict({
            0: 'batch',
            1: 'num_dets'
        }),
        masks=dict({
            0: 'batch',
            1: 'num_dets',
            2: 'height',
            3: 'width'
        })))
codebase_config = dict(
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
        export_postprocess_mask=False))
