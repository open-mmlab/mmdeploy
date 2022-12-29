ir_config = dict(
    type='onnx',
    opset_version=11,
    save_file='end2end.onnx',
)
codebase_config = dict(
    type='mmaction',
    task='VideoRecognition',
    model_type='end2end',
    update_config=True,
    is_dynamic_batch=False)
backend_config = dict(
    type='tensorrt',
    common_config=dict(fp16_mode=False, max_workspace_size=1073741824),
)
