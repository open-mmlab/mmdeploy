# Copyright (c) OpenMMLab. All rights reserved.
import mmdeploy_python as c_api

from mmdeploy.utils import Backend
from ..base import BACKEND_WRAPPER, BaseWrapper


@BACKEND_WRAPPER.register_module(Backend.SDK.value)
class SDKWrapper(BaseWrapper):

    def __init__(self, model_file, task_name, device):
        super().__init__([])
        creator = getattr(c_api, task_name)
        # TODO: get device id somewhere
        self.handle = creator(model_file, device, 0)

    def invoke(self, imgs):
        return self.handle(imgs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('')
