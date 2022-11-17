_base_ = ['./text-detection_dynamic.py', '../../_base_/backends/tensorrt.py']
onnx_config = dict(
    input_shape=(640, 320),
    output_names=['dets', 'labels', 'masks'],
)

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 320, 640],
                    opt_shape=[1, 3, 320, 640],
                    max_shape=[1, 3, 320, 640])))
    ])

codebase_config = dict(
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1))
