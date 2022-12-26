_base_ = ['./text-detection_mrcnn_tensorrt_dynamic-320x320-2240x2240.py']
backend_config = dict(common_config=dict(fp16_mode=True))
