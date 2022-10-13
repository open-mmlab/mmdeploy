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
                 output_names: Optional[Sequence[str]] = None):
        super().__init__(output_names)

        # Create model runner
        config = model_runtime.ModelRunnerConfig()
        config.device_wait_config = model_runtime.DeviceWaitConfig(
            model_runtime.DeviceWaitStrategy.WAIT_WITH_TIMEOUT,
            timeout=timedelta(seconds=600),
            sleepTime=timedelta(seconds=1))

        print("Creating ModelRunner with", config)
        self.runner = model_runtime.ModelRunner(model_runtime.PopefPaths(popef_file),
                                                config=config)

        print("Preparing input tensors:")
        input_descriptions = self.runner.getExecuteInputs()
        input_tensors = [
            np.random.randn(*input_desc.shape).astype(
                popef.popefTypeToNumpyDType(input_desc.data_type))
            for input_desc in input_descriptions
        ]
        self.input_view = model_runtime.InputMemoryView()

        for input_desc, input_tensor in zip(input_descriptions, input_tensors):
            print("\tname:", input_desc.name, "shape:", input_tensor.shape,
                  "dtype:", input_tensor.dtype)
            self.input_view[input_desc.name] = input_tensor

    def forward(self, inputs):

        for key in inputs.keys():
            self.input_view[key] = inputs[key]
        result = self.runner.execute(self.input_view)
        output_descriptions = self.runner.getExecuteOutputs()

        outputs = {}
        print("Processing output tensors:")
        for output_desc in output_descriptions:
            output_tensor = np.frombuffer(result[output_desc.name],
                                          dtype=popef.popefTypeToNumpyDType(
                output_desc.data_type)).reshape(
                output_desc.shape)
            outputs[output_desc.name] = output_tensor
            # print("\tname:", output_desc.name, "shape:", output_tensor.shape,
            #       "dtype:", output_tensor.dtype, "\n", output_tensor)

        return outputs


if __name__ == "__main__":
    main()
