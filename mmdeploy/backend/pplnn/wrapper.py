# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import numpy as np
import onnx
import pyppl.common as pplcommon
import pyppl.nn as pplnn
import torch

from mmdeploy.utils import Backend, parse_device_id
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper
from .utils import register_engines


@BACKEND_WRAPPER.register_module(Backend.PPLNN.value)
class PPLNNWrapper(BaseWrapper):
    """PPLNN wrapper for inference.

    Args:
        onnx_file (str): Path of input ONNX model file.
        algo_file (str): Path of PPLNN algorithm file.
        device_id (int): Device id to put model.

    Examples:
        >>> from mmdeploy.backend.pplnn import PPLNNWrapper
        >>> import torch
        >>>
        >>> onnx_file = 'model.onnx'
        >>> model = PPLNNWrapper(onnx_file, 'end2end.json', 0)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self,
                 onnx_file: str,
                 algo_file: str,
                 device: str,
                 output_names: Optional[Sequence[str]] = None,
                 **kwargs):

        # enable quick select by default to speed up pipeline
        # TODO: open it to users after pplnn supports saving serialized models

        # TODO: assert device is gpu
        device_id = parse_device_id(device)

        # enable quick select by default to speed up pipeline
        # TODO: disable_avx512 will be removed or open to users in config
        engines = register_engines(
            device_id,
            disable_avx512=False,
            quick_select=False,
            import_algo_file=algo_file)
        runtime_builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(
            onnx_file, engines)
        assert runtime_builder is not None, 'Failed to create '\
            'OnnxRuntimeBuilder.'

        runtime = runtime_builder.CreateRuntime()
        assert runtime is not None, 'Failed to create the instance of Runtime.'

        self.runtime = runtime
        self.inputs = {
            runtime.GetInputTensor(i).GetName(): runtime.GetInputTensor(i)
            for i in range(runtime.GetInputCount())
        }

        if output_names is None:
            model = onnx.load(onnx_file)
            output_names = [node.name for node in model.graph.output]

        super().__init__(output_names)

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): Input name and tensor pairs.

        Return:
            Dict[str, torch.Tensor]: The output name and tensor pairs.
        """
        for name, input_tensor in inputs.items():
            input_tensor = input_tensor.contiguous()
            self.inputs[name].ConvertFromHost(input_tensor.cpu().numpy())
        self.__pplnn_execute()
        outputs = {}
        for i in range(self.runtime.GetOutputCount()):
            out_tensor = self.runtime.GetOutputTensor(i).ConvertToHost()
            name = self.output_names[i]
            if out_tensor:
                outputs[name] = np.array(out_tensor, copy=False)
            else:
                out_shape = self.runtime.GetOutputTensor(
                    i).GetShape().GetDims()
                outputs[name] = np.random.rand(*out_shape)
            outputs[name] = torch.from_numpy(outputs[name])
        return outputs

    @TimeCounter.count_time()
    def __pplnn_execute(self):
        """Run inference with PPLNN."""
        status = self.runtime.Run()
        assert status == pplcommon.RC_SUCCESS, 'Run() failed: ' + \
            pplcommon.GetRetCodeStr(status)
