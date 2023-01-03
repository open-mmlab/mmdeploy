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
import time


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
        >>> popef_file = 'resnet.popef'
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
        print('init ipu backend with bps ', bps,
              ' output_names ', output_names)
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
        self.input_view = model_runtime.InputMemoryView()

    @TimeCounter.count_time(Backend.IPU.value)
    def forward(self, inputs):
        start = time.time()
        for input_desc in self.input_descriptions:
            astart = time.time()
            self.input_view[input_desc.name] = inputs[input_desc.name].contiguous().numpy().astype(
                popef.popefTypeToNumpyDType(input_desc.data_type))
            # print('input assign time ', time.time()-astart)
        result = self.runner.executeAsync(self.input_view)
        result.wait()
        # print('ipu forward time ', time.time()-start)
        output_descriptions = self.runner.getExecuteOutputs()

        outputs = {}

        for output_desc in output_descriptions:

            out_shape = output_desc.shape

            out_shape[0] = out_shape[0] * self.bps

            output_tensor = np.frombuffer(result[output_desc.name],
                                          dtype=popef.popefTypeToNumpyDType(
                output_desc.data_type)).reshape(out_shape)

            outputs[output_desc.name] = torch.from_numpy(output_tensor)

        return outputs


if __name__ == "__main__":
    main()
