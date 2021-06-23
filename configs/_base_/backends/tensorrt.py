import tensorrt as trt

backend = 'tensorrt'
tensorrt_param = dict(
    log_level=trt.Logger.WARNING,
    fp16_mode=False,
    save_file='onnx2tensorrt.engine')
