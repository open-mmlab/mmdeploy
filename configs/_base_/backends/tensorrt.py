import tensorrt as trt

backend = 'tensorrt'
tensorrt_param = dict(
    shared_param=dict(log_level=trt.Logger.WARNING, fp16_mode=False))
