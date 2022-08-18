# Copyright (c) OpenMMLab. All rights reserved.
import mmdeploy_python as c_api

from mmdeploy.utils import Backend
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper


@BACKEND_WRAPPER.register_module(Backend.SDK.value)
class SDKWrapper(BaseWrapper):

    def __init__(self, model_file: str, task_name: str, device: str):
        super().__init__([])
        creator = getattr(c_api, task_name)
        device_id = 0
        name_idx = device.split(':')
        if len(name_idx) == 2:
            device, device_id = name_idx
        self.handle = creator(model_file, device, int(device_id))

    @TimeCounter.count_time(Backend.SDK.value)
    def invoke(self, imgs):
        return self.handle(imgs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('')
