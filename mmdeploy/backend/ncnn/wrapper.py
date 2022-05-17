# Copyright (c) OpenMMLab. All rights reserved.
import importlib
from typing import Dict, Optional, Sequence

import ncnn
import numpy as np
import torch

from mmdeploy.utils import Backend, get_root_logger
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper


@BACKEND_WRAPPER.register_module(Backend.NCNN.value)
class NCNNWrapper(BaseWrapper):
    """ncnn wrapper class for inference.

    Args:
        param_file (str): Path of a parameter file.
        bin_file (str): Path of a binary file.
        output_names (Sequence[str] | None): Names of model outputs in order.
            Defaults to `None` and the wrapper will load the output names from
            ncnn model.

    Examples:
        >>> from mmdeploy.backend.ncnn import NCNNWrapper
        >>> import torch
        >>>
        >>> param_file = 'model.params'
        >>> bin_file = 'model.bin'
        >>> model = NCNNWrapper(param_file, bin_file)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self,
                 param_file: str,
                 bin_file: str,
                 output_names: Optional[Sequence[str]] = None,
                 use_vulkan: bool = False,
                 **kwargs):

        net = ncnn.Net()
        if importlib.util.find_spec('mmdeploy.backend.ncnn.ncnn_ext'):
            from mmdeploy.backend.ncnn import ncnn_ext
            ncnn_ext.register_mmdeploy_custom_layers(net)
        net.opt.use_vulkan_compute = use_vulkan
        net.load_param(param_file)
        net.load_model(bin_file)

        self._net = net
        if output_names is None:
            assert hasattr(self._net, 'output_names')
            output_names = self._net.output_names()

        super().__init__(output_names)

    @staticmethod
    def get_backend_file_count() -> int:
        """Return the count of backend file(s)

        ncnn needs a .param file and a .bin file. So the count is 2.

        Returns:
            int: The count of required backend file(s).
        """
        return 2

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): Key-value pairs of model inputs.

        Returns:
            Dict[str, torch.Tensor]: Key-value pairs of model outputs.
        """
        input_list = list(inputs.values())
        batch_size = input_list[0].size(0)
        logger = get_root_logger()
        if batch_size > 1:
            logger.warning(
                f'ncnn only support batch_size = 1, but given {batch_size}')
        for input_tensor in input_list[1:]:
            assert input_tensor.size(
                0) == batch_size, 'All tensors should have same batch size'
            assert input_tensor.device.type == 'cpu', \
                'ncnn only supports cpu device'
        # set output names
        output_names = self._output_names
        # create output dict
        outputs = dict([name, [None] * batch_size] for name in output_names)
        # run inference
        for batch_id in range(batch_size):
            # create extractor
            ex = self._net.create_extractor()

            # set inputs
            for name, input_tensor in inputs.items():
                data = input_tensor[batch_id].contiguous()
                data = data.detach().cpu().numpy()
                input_mat = ncnn.Mat(data)
                ex.input(name, input_mat)

            # get outputs
            result = self.__ncnn_execute(
                extractor=ex, output_names=output_names)
            for name in output_names:
                mat = result[name]
                # deal with special case
                if mat.empty():
                    logger.warning(
                        f'The "{name}" output of ncnn model is empty.')
                    continue
                outputs[name][batch_id] = torch.from_numpy(np.array(mat))

        # stack outputs together
        for name, output_tensor in outputs.items():
            if None in output_tensor:
                outputs[name] = None
            else:
                outputs[name] = torch.stack(output_tensor)

        return outputs

    @TimeCounter.count_time()
    def __ncnn_execute(self, extractor: ncnn.Extractor,
                       output_names: Sequence[str]) -> Dict[str, ncnn.Mat]:
        """Run inference with ncnn.

        Args:
            extractor (ncnn.Extractor): ncnn extractor to extract output.
            output_names (Iterable[str]): A list of string specifying
                output names.

        Returns:
            dict[str, ncnn.Mat]: Inference results of ncnn model.
        """
        result = {}
        for name in output_names:
            out_ret, out = extractor.extract(name)
            assert out_ret == 0, f'Failed to extract output : {out}.'
            result[name] = out
        return result
