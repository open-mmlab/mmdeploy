# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Dict, Optional, Sequence, Union

import torch
import tvm
import tvm.contrib.graph_executor as runtime
from tvm.runtime.vm import Executable, VirtualMachine

from mmdeploy.utils import Backend
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper


@BACKEND_WRAPPER.register_module(Backend.TVM.value)
class TVMWrapper(BaseWrapper):
    """TVM runtime wrapper.

    Args:
        lib (str): The path to the generated lib
        output_names (Sequence[str]): The output names.
        bytecode (Union[bytearray, str]): The bytecode for virtual machine.
        device (str): Device used to do the the inference


    Examples:
        >>> from mmdeploy.backend.tvm import TVMWrapper
        >>> lib_file = 'resnet.so'
        >>> model = TVMWrapper(lib_file, ['output'])
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self,
                 lib: str,
                 output_names: Sequence[str],
                 bytecode: Optional[Union[bytearray, str]] = None,
                 device: str = 'cpu'):
        super().__init__(output_names)
        self.use_vm = False

        if isinstance(lib, str):
            lib = tvm.runtime.load_module(lib)

        match_result = re.match('([^:]+)(:[0-9]+)?$', device)
        assert match_result is not None, f'Can not parse device {device}.'
        device_type = match_result.group(1).lower()
        device_id = 0 if match_result.lastindex == 1 else int(
            match_result.group(2)[1:])
        device = tvm.device(device_type, device_id)

        if bytecode is not None:
            self.use_vm = True
            if isinstance(bytecode, str):
                with open(bytecode, 'rb') as f:
                    bytecode = f.read()

        if self.use_vm:
            exec = Executable.load_exec(bytecode, lib)

            module = VirtualMachine(exec, device)
        else:
            module = runtime.GraphModule(lib['default'](device))
            num_output = module.get_num_outputs()
            assert isinstance(output_names, Sequence)
            assert len(output_names) == num_output

        self._lib = lib
        self._device = device
        self._module = module

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): The input name and tensor pairs.

        Return:
            Dict[str, torch.Tensor]: The output name and tensor pairs.
        """
        module = self._module
        device = self._device

        mod_inputs = dict()
        for name, tensor in inputs.items():
            if tensor.device.type == 'cuda':
                mod_inputs[name] = tvm.nd.from_dlpack(tensor)
            else:
                mod_inputs[name] = tvm.nd.array(tensor.cpu().numpy(), device)

        if self.use_vm:
            module.set_input('main', **mod_inputs)
            self.__tvm_execute()
            vm_ret = module.get_outputs()
            ret = dict()
            for idx, name in enumerate(self._output_names):
                ndarray = vm_ret[idx]
                tensor = torch.from_dlpack(ndarray.to_dlpack())
                ret[name] = tensor
            return ret

        else:
            module.set_input(**mod_inputs)

            self.__tvm_execute()

            ret = dict()
            for idx, name in enumerate(self._output_names):
                ndarray = module.get_output(idx)
                tensor = torch.from_dlpack(ndarray.to_dlpack())
                ret[name] = tensor.clone()
            return ret

    @TimeCounter.count_time(Backend.TVM.value)
    def __tvm_execute(self):
        module = self._module
        module.run()
