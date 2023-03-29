# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.utils import Backend, parse_device_id, parse_device_type
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper


@BACKEND_WRAPPER.register_module(Backend.SDK.value)
class SDKWrapper(BaseWrapper):

    def __init__(self, model_file, task_name, device):
        super().__init__([])
        import mmdeploy_runtime as c_api
        creator = getattr(c_api, task_name)
        device_id = parse_device_id(device)
        device_type = parse_device_type(device)
        # sdk does not support -1 device id
        device_id = 0 if device_id < 0 else device_id
        self.handle = creator(model_file, device_type, device_id)

    @TimeCounter.count_time(Backend.SDK.value)
    def invoke(self, imgs):
        return self.handle(imgs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('')
