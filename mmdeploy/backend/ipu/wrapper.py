# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Any, Dict, Optional, Sequence, Union

import torch
import onnx

from mmdeploy.utils import Backend
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper

import argparse
from datetime import timedelta
import numpy as np
import model_runtime
import popef


@BACKEND_WRAPPER.register_module(Backend.IPU.value)
class IPUWrapper(BaseWrapper):
    """IPU engine wrapper for inference.

    Args:
        ONNX : onnx file for engine to wrap.
        output_names (Sequence[str] | None): Names of model outputs  in order.
            Defaults to `None` and the wrapper will load the output names from
            model.

    Note:
        If the engine is converted from onnx model. The input_names and
        output_names should be the same as onnx model.

    Examples:
        >>> from mmdeploy.backend.ipu import IPUWrapper
        >>> popef_file = 'resnet.ef'
        >>> model = IPUWrapper(popef_file)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self,
                 popef_file: str,
                 bps=1,
                 output_names: Optional[Sequence[str]] = None):
        super().__init__(output_names)
        self.bps = bps
        # Create model runner
        config = model_runtime.ModelRunnerConfig()
        config.device_wait_config = model_runtime.DeviceWaitConfig(
            model_runtime.DeviceWaitStrategy.WAIT_WITH_TIMEOUT,
            timeout=timedelta(seconds=600),
            sleepTime=timedelta(seconds=1))

        print("Creating ModelRunner with", config)
        print("popef path ", popef_file)
        self.runner = model_runtime.ModelRunner(popef_file,
                                                config=config)

        print("Preparing input tensors:")
        self.input_descriptions = self.runner.getExecuteInputs()
        input_tensors = [
            np.random.randn(*input_desc.shape).astype(
                popef.popefTypeToNumpyDType(input_desc.data_type))
            for input_desc in self.input_descriptions
        ]
        # self.input_view = model_runtime.InputMemoryView()

        # for input_desc, input_tensor in zip(self.input_descriptions, input_tensors):
        #     print("\tname:", input_desc.name, "shape:", input_tensor.shape,
        #           "dtype:", input_tensor.dtype)
        #     input_tensor = np.repeat(input_tensor, repeats=self.bps, axis=0)
        #     print('init input tensor ', input_tensor)
        #     self.input_view[input_desc.name] = input_tensor

    def forward(self, inputs):
        input_view = model_runtime.InputMemoryView()
        # print('input desc ', self.input_descriptions)
        for input_desc in self.input_descriptions:
            # print("\tname:", input_desc.name, "shape:", input_tensor.shape,
            #       "dtype:", input_tensor.dtype)
            # input_tensor = np.repeat(input_tensor, repeats=self.bps, axis=0)
            # print('forward input tensor ', inputs[input_desc.name].dtype)
            input_view[input_desc.name] = inputs[input_desc.name].numpy().astype(
                popef.popefTypeToNumpyDType(input_desc.data_type))
            # print('actual input shape ', inputs[input_desc.name].numpy().shape)
            # .astype(popef.popefTypeToNumpyDType(input_desc.data_type))
        # print('input view key ', self.input_view.keys())
        # print('inputs key ', inputs.keys())
        # for key in inputs.keys():
        #     print('input view key val ', key, self.input_view[key])
        #     print('input key val ', key, inputs[key])
        #     print('input view data and type ', type(
        #         self.input_view[key].data), self.input_view[key].data)
        #     self.input_view[key].data = inputs[key]
        result = self.runner.executeAsync(input_view)
        result.wait()
        output_descriptions = self.runner.getExecuteOutputs()
        # print('output desc ', output_descriptions)

        outputs = {}
        # print("Processing output tensors:")
        for output_desc in output_descriptions:

            out_shape = output_desc.shape
            # print('out desc type & shape ', output_desc.data_type, out_shape)
            # out_shape[0] = out_shape[0] * self.bps
            # print(result[output_desc.name], type(
            #     result[output_desc.name]))
            # print('buffer dtype ', popef.popefTypeToNumpyDType(
            #     output_desc.data_type))
            output_tensor = np.frombuffer(result[output_desc.name],
                                          dtype=popef.popefTypeToNumpyDType(
                output_desc.data_type)).reshape(output_desc.shape)
            # print('output tensor ', output_tensor, len(
            #     output_tensor), type(output_tensor[0]))
            # .reshape(output_desc.shape)
            outputs[output_desc.name] = torch.from_numpy(output_tensor)
            # print("\tname:", output_desc.name, "shape:", output_tensor.shape,
            #       "dtype:", output_tensor.dtype, "\n", output_tensor)

        return outputs


if __name__ == "__main__":
    main()
