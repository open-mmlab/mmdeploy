# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

import torch

from mmdeploy.utils import Backend
from mmdeploy.utils.timer import TimeCounter

from ..base import BACKEND_WRAPPER, BaseWrapper


@BACKEND_WRAPPER.register_module(Backend.TORCHSCRIPT.value)
class TorchscriptWrapper(BaseWrapper):
    """Torchscript engine wrapper for inference.

    Args:
        model (torch.jit.RecursiveScriptModule): torchscript engine to wrap.
        input_names (Sequence[str] | None): Names of model inputs  in order.
            Defaults to `None` and the wrapper will accept list or Tensor.
        output_names (Sequence[str] | None): Names of model outputs  in order.
            Defaults to `None` and the wrapper will return list or Tensor.

    Note:
        If the engine is converted from onnx model. The input_names and
        output_names should be the same as onnx model.

    Examples:
        >>> from mmdeploy.backend.torchscript import TorchscriptWrapper
        >>> engine_file = 'resnet.engine'
        >>> model = TorchscriptWrapper(engine_file, input_names=['input'], \
        >>>    output_names=['output'])
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self,
                 model: Union[str, torch.jit.RecursiveScriptModule],
                 input_names: Optional[Sequence[str]] = None,
                 output_names: Optional[Sequence[str]] = None):
        super().__init__(output_names)
        self.ts_model = model
        if isinstance(self.ts_model, str):
            self.ts_model = torch.jit.load(self.ts_model)

        assert isinstance(self.ts_model, torch.jit.RecursiveScriptModule
                          ), 'failed to load torchscript model.'

        self._input_names = input_names
        self._output_names = output_names

    def forward(
        self, inputs: Union[torch.Tensor, Sequence[torch.Tensor],
                            Dict[str, torch.Tensor]]
    ) -> Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]]:
        """Run forward inference.

        Args:
            inputs: The input name and tensor pairs.

        Return:
            outputs: The output name and tensor pairs.
        """

        is_dict_inputs = isinstance(inputs, Dict)
        if is_dict_inputs:
            # inputs to dict
            assert self._input_names is not None, \
                'input names have not been given.'
            inputs = [inputs[input_name] for input_name in self._input_names]
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]

        outputs = self.__torchscript_execute(inputs)

        if self._output_names is not None and is_dict_inputs:
            # output to dict
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
            outputs = dict(zip(self._output_names, outputs))

        return outputs

    @TimeCounter.count_time()
    def __torchscript_execute(self, inputs: Sequence[int]):
        """Run inference with TensorRT.

        Args:
            bindings (list[int]): A list of integer binding the input/output.
        """
        return self.ts_model(*inputs)
