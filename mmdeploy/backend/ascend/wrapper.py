# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Dict, Optional, Sequence, Union

import acl
import torch

from mmdeploy.utils import Backend
from ..base import BACKEND_WRAPPER, BaseWrapper
from operator import mul
from functools import reduce
import numpy as np
from itertools import chain


@BACKEND_WRAPPER.register_module(Backend.ASCEND.value)
class AscendWrapper(BaseWrapper):

    def __init__(self, model: str):

        self.context = AscendWrapper._init_context(0)

        self._model_id, ret = acl.mdl.load_from_file(model)
        assert ret == 0, f'acl.mdl.load_from_file failed: {str(ret)}'

        self._model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self._model_desc, self._model_id)
        assert ret == 0, f'acl.mdl.get_desc failed: {str(ret)}'

        self._dynamic_tensor_index, ret = acl.mdl.get_input_index_by_name(
            self._model_desc, 'ascend_mbatch_shape_data')
        if ret != 0:
            self._dynamic_tensor_index = -1

        self._input_dataset = acl.mdl.create_dataset()
        self._input_buffers = []
        self._input_dims = []

        self._input_size = acl.mdl.get_num_inputs(self._model_desc)
        for input_index in range(self._input_size):
            if input_index == self._dynamic_tensor_index:
                pass
            dims, ret = acl.mdl.get_input_dims(self._model_desc, input_index)
            assert ret == 0, f'acl.mdl.get_input_dims failed: {str(ret)}'
            print(dims)
            self._input_dims.append(dims)
            buffer = self._create_buffer(dims['dims'])
            self._input_buffers.append(buffer)
            _, ret = acl.mdl.add_dataset_buffer(self._input_dataset, buffer)
            assert ret == 0, f'acl.mdl.add_dataset_buffer failed: {str(ret)}'

        self._output_dataset = acl.mdl.create_dataset()
        self._output_buffers = []
        self._output_dims = []

        self._output_size = acl.mdl.get_num_outputs(self._model_desc)
        output_names = []
        for output_index in range(self._output_size):
            dims, ret = acl.mdl.get_output_dims(self._model_desc, output_index)
            assert ret == 0, f'acl.mdl.get_output_dims failed: {str(ret)}'
            dims['name'] = dims['name'].split(':')[-1]
            output_names.append(dims['name'])
            print(dims)
            self._output_dims.append(dims)
            buffer = self._create_buffer(dims['dims'])
            self._output_buffers.append(buffer)
            _, ret = acl.mdl.add_dataset_buffer(self._output_dataset, buffer)
            assert ret == 0, f'acl.mdl.add_dataset_buffer failed: {str(ret)}'

        super().__init__(output_names)

    def _create_buffer(self, dims):
        size = reduce(mul, dims) * 4  # float32 only
        ptr, ret = acl.rt.malloc(size, 0)
        buf = acl.create_data_buffer(ptr, size)
        assert ret == 0, f'acl.rt.malloc failed: {str(ret)}'
        return buf

    @classmethod
    def _init_context(cls, device_id):
        if hasattr(cls, 'context'):
            return cls.context
        ret = acl.init()
        assert ret == 0, f'acl.init failed: {str(ret)}'
        ret = acl.rt.set_device(device_id)
        assert ret == 0, f'acl.rt.set_device failed: {str(ret)}'
        cls.context, ret = acl.rt.create_context(device_id)
        assert ret == 0, f'acl.rt.create_context failed: {str(ret)}'
        return cls.context

    def _get_input_index(self, name):
        index, ret = acl.mdl.get_input_index_by_name(self._model_desc, name)
        assert ret == 0, f'acl.mdl.get_input_index_by_name failed: {str(ret)}'
        return index

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        for name, tensor in inputs.items():
            index = self._get_input_index(name)
            buffer_data = acl.get_data_buffer_addr(self._input_buffers[index])
            buffer_size = acl.get_data_buffer_size(self._input_buffers[index])
            tensor = tensor.contiguous().cpu().numpy().astype(dtype=np.float32)
            assert list(tensor.shape) == self._input_dims[index]['dims']
            ptr, _ = tensor.__array_interface__['data']
            ret = acl.rt.memcpy(buffer_data, buffer_size, ptr, tensor.nbytes,
                                1)
            assert ret == 0, f'acl.rt.memcpy failed: {str(ret)}'

        ret = acl.mdl.execute(self._model_id, self._input_dataset,
                              self._output_dataset)
        assert ret == 0, f'acl.mdl.execute failed: {str(ret)}'

        outputs = {}
        for index, name in enumerate(self._output_names):
            buffer_data = acl.get_data_buffer_addr(self._output_buffers[index])
            buffer_size = acl.get_data_buffer_size(self._output_buffers[index])
            tensor = np.empty(
                self._output_dims[index]['dims'], dtype=np.float32)
            ptr, _ = tensor.__array_interface__['data']
            ret = acl.rt.memcpy(ptr, tensor.nbytes, buffer_data, tensor.nbytes,
                                2)
            assert ret == 0, f'acl.rt.memcpy failed: {str(ret)}'
            outputs[name] = torch.from_numpy(tensor)

        return outputs

    def __del__(self):
        for buffer in chain(self._input_buffers, self._output_buffers):
            data = acl.get_data_buffer_addr(buffer)
            acl.rt.free(data)
            acl.destroy_data_buffer(buffer)
        acl.mdl.destroy_dataset(self._input_dataset)
        acl.mdl.destroy_dataset(self._output_dataset)

        acl.mdl.destroy_desc(self._model_desc)
        acl.mdl.unload(self._model_id)