# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict

import coremltools as ct
import numpy as np
import torch

from mmdeploy.utils import Backend
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper


@BACKEND_WRAPPER.register_module(Backend.COREML.value)
class CoreMLWrapper(BaseWrapper):
    """CoreML wrapper class for inference.

    Args:
        model_file (str): Path of a mlpackage file.
        bin_file (str): Path of a binary file.

    Examples:
        >>> from mmdeploy.backend.coreml import CoreMLWrapper
        >>> import torch
        >>>
        >>> model_file = 'model.mlpackage'
        >>> model = CoreMLWrapper(model_file)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self, model_file: str):
        self.model = ct.models.model.MLModel(
            model_file, compute_units=ct.ComputeUnit.ALL)
        spec = self.model.get_spec()
        output_names = [out.name for out in spec.description.output]
        super().__init__(output_names)

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): Key-value pairs of model inputs.

        Returns:
            Dict[str, torch.Tensor]: Key-value pairs of model outputs.
        """
        model_inputs = dict(
            (k, v.detach().cpu().numpy()) for k, v in inputs.items())
        output = self.__execute(model_inputs)
        for name, tensor in output.items():
            output[name] = torch.from_numpy(tensor)
        return output

    @TimeCounter.count_time(Backend.COREML.value)
    def __execute(self, inputs: Dict[str, np.ndarray]) -> Dict:
        """Run inference with CoreML.

        Args:
            inputs (Dict[str, np.ndarray]): Input data with keys.

        Returns:
            Dict[str, np.ndarray]: Inference results with keys.
        """
        return self.model.predict(inputs)
