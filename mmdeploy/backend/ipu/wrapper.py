# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from datetime import timedelta
from typing import Optional, Sequence

import model_runtime
import numpy as np
import popef
import torch

from mmdeploy.utils import Backend, get_root_logger
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper


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
        self.logger = get_root_logger()
        self.logger.info(
            f'init ipu backend with bps {bps} output_names {output_names}')
        self.bps = bps
        # Create model runner
        config = model_runtime.ModelRunnerConfig()
        config.device_wait_config = model_runtime.DeviceWaitConfig(
            model_runtime.DeviceWaitStrategy.WAIT_WITH_TIMEOUT,
            timeout=timedelta(seconds=600),
            sleepTime=timedelta(seconds=1))

        self.runner = model_runtime.ModelRunner(popef_file, config=config)

        self.input_descriptions = self.runner.getExecuteInputs()
        self.input_view = model_runtime.InputMemoryView()

    def forward(self, inputs):
        for input_desc in self.input_descriptions:
            self.input_view[input_desc.name] = inputs[
                input_desc.name].contiguous().numpy().astype(
                    popef.popefTypeToNumpyDType(input_desc.data_type))

        result = self.__ipu_execute()

        output_descriptions = self.runner.getExecuteOutputs()
        outputs = {}
        for output_desc in output_descriptions:
            out_shape = output_desc.shape
            out_shape[0] = out_shape[0] * self.bps
            output_tensor = np.frombuffer(
                result[output_desc.name],
                dtype=popef.popefTypeToNumpyDType(
                    output_desc.data_type)).reshape(out_shape)
            outputs[output_desc.name] = torch.from_numpy(output_tensor)

        return outputs

    @TimeCounter.count_time(Backend.IPU.value)
    def __ipu_execute(self):
        """Run inference with ipu.
        Args:

        Returns:
            dict[str, tensor]: Inference results of ipu model.
        """

        result = self.runner.executeAsync(self.input_view)
        result.wait()
        return result
