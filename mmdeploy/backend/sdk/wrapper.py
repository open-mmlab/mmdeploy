# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Optional, Sequence, Union

import torch

from mmdeploy.utils import Backend
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper
import mmdeploy_python as c_api

api_wrapper_creators = dict(
    classification=getattr(c_api, 'Classifier', None),
    detection=getattr(c_api, 'Detector', None),
    segmentation=getattr(c_api, 'Segmentor', None),
    text_detection=getattr(c_api, 'TextDetector', None),
    text_recognition=getattr(c_api, 'TextRecognizer', None),
    restoration=getattr(c_api, 'Restorer', None))


@BACKEND_WRAPPER.register_module(Backend.SDK.value)
class SDKWrapper(BaseWrapper):

    def __init__(self, model_file, task_name, device):
        super().__init__([])
        creator = api_wrapper_creators[task_name]
        self.handle = creator(model_file, device, 0)

    @staticmethod
    def get_backend_file_count() -> int:
        return 2

    def invoke(self, imgs):
        return self.handle(imgs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("")
