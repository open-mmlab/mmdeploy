_base_ = ['./text-detection_mrcnn_tensorrt_dynamic-320x320-2240x2240.py']

backend_config = dict(common_config=dict(fp16_mode=True, int8_mode=True))

calib_config = dict(create_calib=True, calib_file='calib_data.h5')
