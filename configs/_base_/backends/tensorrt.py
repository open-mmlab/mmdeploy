import tensorrt as trt

backend = 'tensorrt'
tensorrt_params = dict(
    shared_params=dict(fp16_mode=False, log_level=trt.Logger.INFO))
