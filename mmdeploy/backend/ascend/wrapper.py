# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

import acl
import torch

from mmdeploy.utils import Backend
from ..base import BACKEND_WRAPPER, BaseWrapper
from operator import mul
from functools import reduce
import numpy as np
from itertools import chain
from typing import NamedTuple

_from_acl_data_type = {0: np.float32, 3: np.int32}

_to_acl_data_type = {np.float32: 0}


class AclError(Exception):
    pass


def _check(code: int, msg: str):
    if code != 0:
        raise AclError(msg, code)


class DataBuffer:

    def __init__(self, size):
        data, ret = acl.rt.malloc(size, 0)
        _check(ret, 'acl.rt.malloc')
        self.data = data
        self.size = size
        self.handle = acl.create_data_buffer(data, size)

    def __del__(self):
        acl.destroy_data_buffer(self.handle)
        acl.rt.free(self.data)


class Dataset:

    def __init__(self):
        self.handle = acl.mdl.create_dataset()
        self.buffers = []

    def __del__(self):
        acl.mdl.destroy_dataset(self.handle)

    def add_buffer(self, buffer: DataBuffer):
        self.buffers.append(buffer)
        _, ret = acl.mdl.add_dataset_buffer(self.handle, buffer.handle)
        _check(ret, 'acl.mdl.add_dataset_buffer')


class Binding(NamedTuple):
    index: int
    name: str
    dims: List[int]
    data_type: np.dtype
    size: int


class ModelDesc:

    def __init__(self, model_id):
        self._desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self._desc, model_id)
        _check(ret, 'acl.mdl.get_desc')

        self.inputs = []
        self.dynamic_tensor = None
        num_inputs = acl.mdl.get_num_inputs(self._desc)
        for index in range(num_inputs):
            dims = self._get_input_dims(index)
            data_type = acl.mdl.get_input_data_type(self._desc, index)
            data_type = _from_acl_data_type[data_type]
            size = acl.mdl.get_input_size_by_index(self._desc, index)
            binding = Binding(index, dims['name'], dims['dims'], data_type,
                              size)
            if dims['name'] == 'ascend_mbatch_shape_data':
                self.dynamic_tensor = binding
            else:
                self.inputs.append(binding)

        self.outputs = []
        num_outputs = acl.mdl.get_num_outputs(self._desc)
        for index in range(num_outputs):
            dims = self._get_output_dims(index)
            data_type = acl.mdl.get_output_data_type(self._desc, index)
            data_type = _from_acl_data_type[data_type]
            size = acl.mdl.get_input_size_by_index(self._desc, index)
            self.outputs.append(
                Binding(index, dims['name'], dims['dims'], data_type, size))

    def __del__(self):
        acl.mdl.destroy_desc(self._desc)

    def _get_input_dims(self, index):
        dims, ret = acl.mdl.get_input_dims(self._desc, index)
        _check(ret, 'acl.mdl.get_input_dims')
        return dims

    def _get_output_dims(self, index):
        dims, ret = acl.mdl.get_output_dims(self._desc, index)
        _check(ret, 'acl.mdl.get_output_dims')
        dims['name'] = dims['name'].split(':')[-1]
        return dims

    def _get_current_output_dims(self, index):
        dims, ret = acl.mdl.get_cur_output_dims(self._desc, index)
        _check(ret, 'acl.mdl.get_cur_output_dims')
        return dims

    def get_current_ouptut_dims(self):
        dimses = []
        for output in self.outputs:
            dims = self._get_current_output_dims(output.index)
            dimses.append(dims['dims'])
        return dimses

    def _get_input_index(self, name):
        index, ret = acl.mdl.get_input_index_by_name(self._desc, name)
        return index if ret == 0 else -1

    def get_dynamic_batch(self):
        batch, ret = acl.mdl.get_dynamic_batch(self._desc)
        _check(ret, 'acl.mdl.get_dynamic_batch')
        batch = batch['batch']
        return sorted(batch)

    def get_dynamic_hw(self):
        hw_info, ret = acl.mdl.get_dynamic_hw(self._desc, -1)
        _check(ret, 'acl.mdl.get_dynamic_hw')
        return hw_info['hw']

    def get_input_dynamic_dims(self):
        count, ret = acl.mdl.get_input_dynamic_gear_count(self._desc, -1)
        _check(ret, 'acl.mdl.get_input_dynamic_gear_count')
        dims, ret = acl.mdl.get_input_dynamic_dims(self._desc, -1, count)
        _check(ret, 'acl.mdl.get_input_dynamic_dims')
        return dims


@BACKEND_WRAPPER.register_module(Backend.ASCEND.value)
class AscendWrapper(BaseWrapper):

    def __init__(self, model: str):

        self.context = AscendWrapper._init_context(0)

        self._model_id, ret = acl.mdl.load_from_file(model)
        _check(ret, 'acl.mdl.load_from_file')

        self._model_desc = ModelDesc(self._model_id)

        self._config_dynamic_shapes()
        self._create_input_buffers()
        self._create_output_buffers()

        output_names = [output.name for output in self._model_desc.outputs]

        super().__init__(output_names)

    def __del__(self):
        acl.mdl.unload(self._model_id)

    @classmethod
    def _init_context(cls, device_id):
        if hasattr(cls, 'context'):
            return cls.context

        _check(acl.init(), 'acl.init')
        _check(acl.rt.set_device(device_id), 'acl.rt.set_device')

        cls.context, ret = acl.rt.create_context(device_id)
        _check(ret, 'acl.rt.create_context')

        return cls.context

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_shapes = [inputs[x.name].shape for x in self._model_desc.inputs]

        output_shapes = self._reshape(input_shapes)

        for binding in self._model_desc.inputs:
            buffer_data = self._input.buffers[binding.index].data
            buffer_size = self._input.buffers[binding.index].size
            tensor = inputs[binding.name].contiguous().cpu().numpy()
            if tensor.dtype != binding.data_type:
                tensor = tensor.astype(binding.data_type)
            ptr, _ = tensor.__array_interface__['data']
            ret = acl.rt.memcpy(buffer_data, buffer_size, ptr, tensor.nbytes,
                                1)
            _check(ret, 'acl.rt.memcpy')

        ret = acl.mdl.execute(self._model_id, self._input.handle,
                              self._output.handle)
        _check(ret, 'acl.mdl.execute')

        outputs = {}
        for binding in self._model_desc.outputs:
            buffer_data = self._output.buffers[binding.index].data
            buffer_size = self._output.buffers[binding.index].size
            tensor = np.empty(
                output_shapes[binding.index], dtype=binding.data_type)
            ptr, _ = tensor.__array_interface__['data']
            ret = acl.rt.memcpy(ptr, tensor.nbytes, buffer_data, tensor.nbytes,
                                2)
            _check(ret, 'acl.rt.memcpy')
            outputs[binding.name] = torch.from_numpy(tensor)

        return outputs

    def _verify_dims(self, src: Sequence[int], ref: Sequence[int]):
        if len(src) != len(ref):
            raise RuntimeError(f'Shape mismatch {src} vs {ref}')
        for src_dim, ref_dim in zip(src, ref):
            if ref_dim != -1 and src_dim != ref_dim:
                raise RuntimeError(f'Shape mismatch {src} vs {ref}')

    def _reshape(self, input_shapes):

        if len(input_shapes) != len(self._model_desc.inputs):
            raise RuntimeError('#inputs mismatch')

        for src, ref in zip(input_shapes, self._model_desc.inputs):
            self._verify_dims(src, ref.dims)

        self._reshape_fn(input_shapes)

        dimses = self._model_desc.get_current_ouptut_dims()
        return dimses

    def _reshape_static(self, input_shapes):
        pass

    def _reshape_dynamic_batch_size(self, input_shapes):
        batch_size = None
        for src, ref in zip(input_shapes, self._model_desc.inputs):
            tmp_batch_size = None
            for src_dim, ref_dim in zip(src, ref.dims):
                if ref_dim == -1:
                    tmp_batch_size = src_dim
            if tmp_batch_size and batch_size is None:
                batch_size = tmp_batch_size
            elif tmp_batch_size and batch_size != tmp_batch_size:
                raise RuntimeError(
                    f'Inconsistent batch size {batch_size} vs {tmp_batch_size}'
                )
        if batch_size is None:
            raise RuntimeError('Can\'t determine batch size')

        candidates = list(
            filter(lambda x: x >= batch_size, self._dynamic_batch_size))
        if not candidates:
            raise RuntimeError(
                f'Batch size {batch_size} is not supported. ({self._dynamic_batch_size})'
            )

        ret = acl.mdl.set_dynamic_batch_size(
            self._model_id, self._input.handle,
            self._model_desc.dynamic_tensor.index, candidates[0])
        _check(ret, 'acl.mdl.set_dynamic_batch_size')

    def _get_hw(self, src: Sequence[int], ref: Sequence[int]) -> Tuple[int]:
        hw = []
        for src_dim, ref_dim in zip(src, ref):
            if ref_dim == -1:
                hw.append(src_dim)
        if not hw:
            return ()
        if len(hw) != 2:
            raise RuntimeError('Can\'t determine HW')
        return tuple(*hw)

    def _reshape_dynamic_image_size(self, input_shapes):
        hw = None
        for src, ref in zip(input_shapes, self._model_desc.inputs):
            tmp_hw = self._get_hw(src, ref.dims)
            if tmp_hw and hw is None:
                hw = tmp_hw
            elif tmp_hw and tmp_hw != hw:
                raise RuntimeError(f'Inconsistent image size {hw} vs {tmp_hw}')
        if hw is None:
            raise RuntimeError('Can\'t determine dynamic HW')
        if not hw in self._dynamic_hw:
            raise RuntimeError(
                f'HW {hw} is not supported. ({self._dynamic_hw})')
        height, width = hw
        ret = acl.mdl.set_dynamic_hw_size(
            self._model_id, self._input.handle,
            self._model_desc.dynamic_tensor.index, height, width)
        _check(ret, 'acl.mdl.set_dynamic_hw_size')

    def _reshape_dynamic_dims(self, input_shapes):
        match = [True] * len(self._dynamic_dims)
        ptr = 0
        for src in input_shapes:
            for axis, src_dim in enumerate(src):
                for index, dims in enumerate(self._dynamic_dims):
                    if axis == 0 and src_dim < dims['dims'][
                            ptr]:  # allow batch dimension to vary
                        pass
                    elif src_dim != dims[ptr]:
                        match[index] = False

        indices = [i for i, v in enumerate(match) if v]
        if not indices:
            raise RuntimeError('No matching profile found')
        index = indices[0]

        ret = acl.mdl.set_input_dynamic_dims(
            self._model_id, self._input.handle,
            self._model_desc.dynamic_tensor.index, self._dynamic_dims[index])
        _check(ret, 'acl.mdl.set_input_dynamic_dims')

    def _config_dynamic_shapes(self):

        if self._model_desc.dynamic_tensor is None:
            self._input_shape_type = 'static'
            self._reshape_fn = self._reshape_static
            return

        self._dynamic_batch_size = self._model_desc.get_dynamic_batch()
        if self._dynamic_batch_size:
            self._input_shape_type = 'dynamic_batch_size'
            self._reshape_fn = self._reshape_dynamic_batch_size
            return

        self._dynamic_dims = self._model_desc.get_input_dynamic_dims()
        if self._dynamic_dims:
            self._input_shape_type = 'dynamic_dims'
            self._reshape_fn = self._reshape_dynamic_image_size
            return

        self._dynamic_hw = self._model_desc.get_dynamic_hw()
        if self._dynamic_hw:
            self._input_shape_type = 'dynamic_image_size'
            self._reshape_fn = self._reshape_dynamic_dims
            return

        raise RuntimeError('Can\'t infer input shape type')

    def _create_input_buffers(self):
        self._input = Dataset()
        for binding in self._model_desc.inputs:
            self._input.add_buffer(DataBuffer(binding.size))
        if self._model_desc.dynamic_tensor:
            self._input.add_buffer(
                DataBuffer(self._model_desc.dynamic_tensor.size))

    def _create_output_buffers(self):
        self._output = Dataset()
        for binding in self._model_desc.outputs:
            self._output.add_buffer(DataBuffer(binding.size))
