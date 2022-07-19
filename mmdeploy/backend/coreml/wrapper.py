# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, Optional, Sequence, Union
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper
from mmdeploy.utils import Backend
import torch
import coremltools as ct


@BACKEND_WRAPPER.register_module(Backend.COREML.value)
class CoreMLWrapper(BaseWrapper):

    def __init__(self,
                 model_file: str):
        self.model = ct.models.model.MLModel(model_file)
        spec = self.model.get_spec()
        output_names = [out.name for out in spec.description.output]
        super().__init__(output_names)

    def forward(self,
                inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self.model.predict(inputs)
        for name, tensor in output.items():
            output[name] = torch.from_numpy(tensor)
        return output

    @TimeCounter.count_time()
    def __execute(self,
                  inputs: Dict[str, torch.Tensor]) -> Dict:
        return self.model.predict(inputs)
