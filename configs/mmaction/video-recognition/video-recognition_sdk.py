_base_ = ['./video-recognition_static.py', '../../_base_/backends/sdk.py']

codebase_config = dict(model_type='sdk')

# It will read SampleFrames from model_cfg to info
# in pipeline of backend_config

backend_config = dict(pipeline=[
    dict(type='OpenCVInit', num_threads=1),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='OpenCVDecode'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ListToNumpy', keys=['imgs'])
])
