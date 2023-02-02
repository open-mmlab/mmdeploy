# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
from typing import Dict, Optional, Sequence, Union

import torch

from mmdeploy.utils import Backend, get_root_logger
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper
from .init_plugins import get_ops_path


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
        logger = get_root_logger()

        # load custom ops if exist
        custom_ops_path = get_ops_path()
        if osp.exists(custom_ops_path):
            torch.ops.load_library(custom_ops_path)

        # import torchvision for ops
        try:
            importlib.import_module('torchvision')
        except Exception:
            logger.warning(
                'Can not import torchvision. '
                'Models require ops in torchvision might not available.')
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
            inputs (torch.Tensor | Sequence[torch.Tensor] | Dict[str,
                torch.Tensor]): The input tensor, or tensor sequence, or pairs
                of input names and tensors.

        Return:
            outputs (torch.Tensor | Sequence[torch.Tensor] | Dict[str,
                torch.Tensor]): The input tensor, or tensor sequence, or pairs
                of input names and tensors.
        """

        is_dict_inputs = isinstance(inputs, Dict)
        if is_dict_inputs:
            # inputs to dict
            assert self._input_names is not None, \
                'input names have not been given.'
            inputs = [inputs[input_name] for input_name in self._input_names]
        elif isinstance(inputs, torch.Tensor):
            inputs = [inputs]

        outputs = self.__torchscript_execute(inputs)

        if self._output_names is not None and is_dict_inputs:
            # output to dict
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
            outputs = dict(zip(self._output_names, outputs))

        if isinstance(outputs, tuple) and self._output_names is not None:
            assert len(outputs) == len(self._output_names)
            outputs = dict(zip(self._output_names, outputs))
        return outputs

    @TimeCounter.count_time(Backend.TORCHSCRIPT.value)
    def __torchscript_execute(
            self, inputs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """Run inference with TorchScript.

        Args:
            inputs (Sequence[torch.Tensor]): A list of integer binding the
            input/output.
        Returns:
            torch.Tensor | Sequence[torch.Tensor]: The inference outputs from
            TorchScript.
        """
        return self.ts_model(*inputs)
