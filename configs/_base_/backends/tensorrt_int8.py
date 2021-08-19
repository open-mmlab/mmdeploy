_base_ = ['./tensorrt.py']

create_calib = True
calib_params = dict(calib_file='calib_data.h5')
tensorrt_params = dict(shared_params=dict(fp16_mode=True, int8_mode=True))
