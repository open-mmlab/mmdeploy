# Copyright (c) OpenMMLab. All rights reserved.
import importlib
from typing import Dict, Optional, Sequence, Union, Iterable, Generator

import json
import torch
import numpy as np
import ctypes
from queue import Queue
from threading import Event, Thread
import vacl_stream
import vaststream

from mmdeploy.utils import Backend, get_root_logger
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper
from mmdeploy.utils import get_onnx_config, get_common_config
from mmdeploy.utils.config_utils import get_backend_config

class VACCForward:
    def __init__(
        self,
        model_info: Union[str, Dict[str, str]],
        vdsp_params_info: str,
        device_id: int = 0,
        batch_size: int = 1,
    ) -> None:
        if isinstance(model_info, str):
            with open(model_info) as f:
                model_info = json.load(f)

        self.device_id = device_id
        self.input_id = 0
        self.vast_stream = vaststream.vast_stream()
        self.input_dict = {}
        self.event_dict = {}
        self.result_dict = {}

        balance_mode = 0

        def callback(output_description, ulOutPointerArray, ulArraySize, user_data_ptr):
            user_data = ctypes.cast(user_data_ptr, ctypes.POINTER(vacl_stream.StreamInfo))
            input_id = output_description.contents.input_id

            device_ddr = self.input_dict.pop(input_id)
            self.vast_stream.free_data_on_device(device_ddr, self.device_id)

            model_name = user_data.contents.model_name
            stream_output_list = self.vast_stream.stream_get_stream_output(model_name, ulOutPointerArray, ulArraySize)
            heatmap = np.squeeze(stream_output_list[0])
            
            num_outputs = self.vast_stream.get_output_num_per_batch(model_name)
            heatmap_shape = []
            for i in range(num_outputs):
                _, shape = self.vast_stream.get_output_shape_by_index(model_name, i)
                ndims = shape.ndims
                _shape = []
                for i in range(ndims):
                    _shape.append(shape.shapes[i])
                heatmap_shape.append(_shape)

            '''
            ndims = shape.ndims
            heatmap_shape = []
            for i in range(ndims):
                heatmap_shape.append(shape.shapes[i])
            '''
            self.result_dict[input_id] = (num_outputs, heatmap_shape, heatmap)
            self.event_dict[input_id].set()

        self.callback = vaststream.output_callback_type(callback)

        self.stream = vacl_stream.create_vaststream(
            device_id,
            vdsp_params_info,
            model_info,
            self.callback,
            balance_mode,
            batch_size,
        )

    def __start_extract(self, image: Union[str, np.ndarray]) -> int:
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        image_size = int(height * width * c)
        device_ddr = self.stream.copy_data_to_device(image, image_size)

        input_id = self.input_id

        self.input_dict[input_id] = device_ddr
        self.event_dict[input_id] = Event()
        self.stream.run_stream_dynamic([device_ddr], (height, width), input_id)
        self.input_id += 1

        return input_id

    def get_output_num(self):
        num_outputs = self.vast_stream.get_output_num_per_batch(self.vast_stream.model_name)
        return num_outputs

    
    def extract(self, image: Union[str, np.ndarray]) -> str:
        input_id = self.__start_extract(image)
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]
        return result

    def extract_batch(self, images: Iterable[Union[str, np.ndarray]]) -> Generator[str, None, None]:
        queue = Queue(20)

        def input_thread():
            for image in images:
                input_id = self.__start_extract(image)
                queue.put(input_id)
            queue.put(None)

        thread = Thread(target=input_thread)
        thread.start()
        while True:
            input_id = queue.get()
            if input_id is None:
                break
            self.event_dict[input_id].wait()
            result = self.result_dict.pop(input_id)
            del self.event_dict[input_id]
            yield result


@BACKEND_WRAPPER.register_module(Backend.VACC.value)
class VACCWrapper(BaseWrapper):
    """vacc wrapper class for inference.

    Args:
        model_info_json (str): Path of a model info json file.
        vdsp_params_info_json (str): Path of a vdsp params info json file.
        output_names (Sequence[str] | None): Names of model outputs in order.
            Defaults to `None` and the wrapper will load the output names from
            vacc model.
    """

    def __init__(self,
                 model_info_json,
                 vdsp_params_info_json,
                 output_names: Optional[Sequence[str]] = None,
                 **kwargs):

        self.model = VACCForward(model_info_json, vdsp_params_info_json)

        super().__init__(output_names)

    @staticmethod
    def get_backend_file_count() -> int:
        """Return the count of backend file(s)

        ncnn needs a .param file and a .bin file. So the count is 2.

        Returns:
            int: The count of required backend file(s).
        """
        return 3

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        input_list = list(inputs.values())
        batch_size = input_list[0].size(0)
        logger = get_root_logger()
        if batch_size > 1:
            logger.warning(
                f'vacc only support batch_size = 1, but given {batch_size}')

        outputs = dict([name, [None] * batch_size] for name in self.output_names)

        for batch_id in range(batch_size):
            output = []
            # set inputs
            for name, input_tensor in inputs.items():
                data = input_tensor[batch_id].contiguous()
                data = data.detach().cpu().numpy()
                results = self.model.extract_batch([data])
                for result in results:
                    output_num = result[0]
                    if output_num == 1:
                        output.append(np.reshape(np.array(result[2]).astype(np.float32), result[1][0])[0])
                    else:
                        outputs_ = {}
                        outputs = {}
                        for index in range(output_num):
                            out = np.reshape(result[2][index].astype(np.float32), result[1][index])
                            outputs_[index] = torch.from_numpy(out)
                        outputs['output'] = outputs_
                        return outputs
            output = np.array(output)
            for name in self.output_names:
                outputs[name][batch_id] = torch.from_numpy(output[0])
        for name, output_tensor in outputs.items():
            if None in output_tensor:
                outputs[name] = None
            else:
                outputs[name] = torch.stack(output_tensor)
        return outputs

