_base_ = ['./rotated-detection_tensorrt_dynamic-320x320-1024x1024.py']

backend_config = dict(common_config=dict(fp16_mode=True))
