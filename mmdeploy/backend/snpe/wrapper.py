# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
from typing import Dict, Optional, Sequence

import grpc
# import mmdeploy.backend.snpe.inference_pb2
# import mmdeploy.backend.snpe.inference_pb2_grpc
import inference_pb2
import inference_pb2_grpc
import numpy as np
import torch

from mmdeploy.utils import Backend, get_root_logger
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper


@BACKEND_WRAPPER.register_module(Backend.SNPE.value)
class SNPEWrapper(BaseWrapper):
    """ncnn wrapper class for inference.

    Args:
        dlc_file (str): Path of a weight file.
        output_names (Sequence[str] | None): Names of model outputs in order.
            Defaults to `None` and the wrapper will load the output names from
            snpe model.

    Examples:
        >>> from mmdeploy.backend.snpe import SNPEWrapper
        >>> import torch
        >>>
        >>> snple_file = 'alexnet.dlc'
        >>> model = SNPEWrapper(snpe_file)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self,
                 dlc_file: str,
                 output_names: Optional[Sequence[str]] = None,
                 **kwargs):

        logger = get_root_logger()

        # The maximum model file size is 512MB
        MAX_SIZE = 2 << 29
        uri = os.environ['__MMDEPLOY_GRPC_URI']
        logger.info(f'fetch uri: {uri}')
        self.channel = grpc.insecure_channel(
            uri,
            options=(('grpc.GRPC_ARG_KEEPALIVE_TIME_MS',
                      2000), ('grpc.max_send_message_length', MAX_SIZE),
                     ('grpc.keepalive_permit_without_calls', 1)))

        weights = bytes()
        filesize = os.stat(dlc_file).st_size

        logger.info(f'reading local model file {dlc_file}')
        # with open(dlc_file, 'rb') as f:
        #     weights = f.read(filesize)

        stub = inference_pb2_grpc.InferenceStub(self.channel)
        logger.info(f'init remote SNPE engine with RPC, please wait...')
        model = inference_pb2.Model(name=dlc_file, weights=weights, device=1)
        resp = stub.Init(model)

        if resp.status != 0:
            logger.error(f'init SNPE model failed {resp.info}')
            return

        output = stub.OutputNames(inference_pb2.Empty())
        output_names = output.names

        super().__init__(output_names)
        logger.info(f'init success, outputs {output_names}')

    def get_shape(self, shape):
        if len(shape) == 4:
            return (0,2,3,1)
        elif len(shape) == 3:
            return (1,2,0)
        elif len(shape) == 2:
            return (0,1)
        return (0)
            
    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): Key-value pairs of model inputs.

        Returns:
            Dict[str, torch.Tensor]: Key-value pairs of model outputs.
        """
        input_list = list(inputs.values())
        device_type = input_list[0].device.type

        logger = get_root_logger()

        # build `list` inputs for remote snpe engine
        snpe_inputs = []
        for name, input_tensor in inputs.items():
            data = input_tensor.contiguous().detach()
            # snpe input layout is  NHWC
            data = data.permute(self.get_shape(data.shape))
            data = data.cpu().numpy()
            
            if data.dtype != np.float32:
                logger.error('SNPE now only support fp32 input')
                data = data.astype(dtype=np.float32)
            tensor = inference_pb2.Tensor(
                data=data.tobytes(), name=name, dtype='float32')

            snpe_inputs.append(tensor)

        return self.__snpe_execute(
            inference_pb2.TensorList(datas=snpe_inputs), device_type)

    @TimeCounter.count_time()
    def __snpe_execute(self, tensorList: inference_pb2.TensorList,
                       device: str) -> Dict[str, torch.tensor]:
        """Run inference with snpe remote inference engine.

        Args:
            tensorList (inference_pb2.TensorList): snpe input tensor.

        Returns:
            dict[str, torch.tensor]: Inference results of snpe model.
        """
        stub = inference_pb2_grpc.InferenceStub(self.channel)
        resp = stub.Inference(tensorList)

        result = dict()
        if resp.status == 0:
            for tensor in resp.datas:
                ndarray = np.frombuffer(tensor.data, dtype=np.float32)
                
                shape = tuple(tensor.shape)
                result[tensor.name] = torch.from_numpy(ndarray.reshape(shape).copy()).to(device)
        else:
            logger = get_root_logger()
            logger.error(f'snpe inference failed {resp.info}')

        return result
