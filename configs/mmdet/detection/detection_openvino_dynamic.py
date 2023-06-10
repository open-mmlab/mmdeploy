ir_config = dict(
    type='onnx',
    opset_version=11,
    save_file='end2end.onnx',
)
codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    model_type='end2end',
    update_config=True,
    is_dynamic_batch=True,
    is_dynamic_size=True,
    detection_mode='detection')
backend_config = dict(type='openvino')
