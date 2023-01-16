_base_ = ['./text-detection_static.py', '../../_base_/backends/tensorrt.py']
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

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 320, 320],
                    opt_shape=[1, 3, 600, 800],
                    max_shape=[1, 3, 2240, 2240])))
    ])

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
