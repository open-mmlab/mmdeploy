# Copyright (c) OpenMMLab. All rights reserved.
import abc
import os
import time
from random import randint
from typing import Dict, Optional, Sequence, Tuple

import grpc
import inference_pb2
import inference_pb2_grpc
import numpy as np
import torch

from mmdeploy.backend.snpe.onnx2dlc import get_env_key
from mmdeploy.utils import Backend, get_root_logger
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper


# add interceptor to sleep and retry request
# https://github.com/grpc/grpc/issues/19514
class SleepingPolicy(abc.ABC):

    @abc.abstractmethod
    def sleep(self, try_i: int):
        """How long to sleep in milliseconds.

        :param try_i: the number of retry (starting from zero)
        """
        assert try_i >= 0


class ExponentialBackoff(SleepingPolicy):

    def __init__(self, *, init_backoff_ms: int, max_backoff_ms: int,
                 multiplier: int):
        self.init_backoff = randint(0, init_backoff_ms)
        self.max_backoff = max_backoff_ms
        self.multiplier = multiplier

    def sleep(self, try_i: int):
        sleep_range = min(self.init_backoff * self.multiplier**try_i,
                          self.max_backoff)
        sleep_ms = randint(0, sleep_range)
        logger = get_root_logger()
        logger.debug(f'Sleeping for {sleep_ms}')
        time.sleep(sleep_ms / 1000)


class RetryOnRpcErrorClientInterceptor(grpc.UnaryUnaryClientInterceptor,
                                       grpc.StreamUnaryClientInterceptor):

    def __init__(
        self,
        *,
        max_attempts: int,
        sleeping_policy: SleepingPolicy,
        status_for_retry: Optional[Tuple[grpc.StatusCode]] = None,
    ):
        self.max_attempts = max_attempts
        self.sleeping_policy = sleeping_policy
        self.status_for_retry = status_for_retry

    def _intercept_call(self, continuation, client_call_details,
                        request_or_iterator):

        for try_i in range(self.max_attempts):
            response = continuation(client_call_details, request_or_iterator)

            if isinstance(response, grpc.RpcError):

                # Return if it was last attempt
                if try_i == (self.max_attempts - 1):
                    return response

                # If status code is not in retryable status codes
                if (self.status_for_retry
                        and response.code() not in self.status_for_retry):
                    return response

                self.sleeping_policy.sleep(try_i)
            else:
                return response

    def intercept_unary_unary(self, continuation, client_call_details,
                              request):
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_stream_unary(self, continuation, client_call_details,
                               request_iterator):
        return self._intercept_call(continuation, client_call_details,
                                    request_iterator)


@BACKEND_WRAPPER.register_module(Backend.SNPE.value)
class SNPEWrapper(BaseWrapper):
    """snpe wrapper class for inference.

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
                 uri: str,
                 output_names: Optional[Sequence[str]] = None,
                 **kwargs):

        logger = get_root_logger()

        interceptors = (RetryOnRpcErrorClientInterceptor(
            max_attempts=4,
            sleeping_policy=ExponentialBackoff(
                init_backoff_ms=100, max_backoff_ms=1600, multiplier=2),
            status_for_retry=(grpc.StatusCode.UNAVAILABLE, ),
        ), )

        if uri is None and get_env_key() in os.environ:
            logger.warn(
                'snpe remote service URI not set, search from environment')
            uri = os.environ[get_env_key()]

        if uri is None:
            logger.error('URI not set')

        weights = bytes()
        filesize = os.stat(dlc_file).st_size

        logger.info(f'reading local model file {dlc_file}')
        with open(dlc_file, 'rb') as f:
            weights = f.read(filesize)

        self.stub = inference_pb2_grpc.InferenceStub(
            grpc.intercept_channel(grpc.insecure_channel(uri), *interceptors))

        logger.info('init remote SNPE engine with RPC, please wait...')
        model = inference_pb2.Model(name=dlc_file, weights=weights, device=1)
        resp = self.stub.Init(model)

        if resp.status != 0:
            logger.error(f'init SNPE model failed {resp.info}')
            return

        output = self.stub.OutputNames(inference_pb2.Empty())
        output_names = output.names

        super().__init__(output_names)
        logger.info(f'init success, outputs {output_names}')

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): Key-value pairs of model inputs.

        Returns:
            Dict[str, torch.Tensor]: Key-value pairs of model outputs.
        """

        def get_shape(shape):
            if len(shape) == 4:
                return (0, 2, 3, 1)
            elif len(shape) == 3:
                return (0, 1, 2)
            elif len(shape) == 2:
                return (0, 1)
            return (0)

        input_list = list(inputs.values())
        device_type = input_list[0].device.type

        logger = get_root_logger()

        # build `list` inputs for remote snpe engine
        snpe_inputs = []
        for name, input_tensor in inputs.items():
            data = input_tensor.contiguous().detach()
            # snpe input layout is  NHWC
            data = data.permute(get_shape(data.shape))
            data = data.cpu().numpy()

            if data.dtype != np.float32:
                logger.error('SNPE now only support fp32 input')
                data = data.astype(dtype=np.float32)
            tensor = inference_pb2.Tensor(
                data=data.tobytes(),
                name=name,
                dtype='float32',
                shape=list(data.shape))

            snpe_inputs.append(tensor)

        return self.__snpe_execute(
            tensorList=inference_pb2.TensorList(data=snpe_inputs),
            device=device_type)

    @TimeCounter.count_time(Backend.SNPE.value)
    def __snpe_execute(self, tensorList: inference_pb2.TensorList,
                       device: str) -> Dict[str, torch.tensor]:
        """Run inference with snpe remote inference engine.

        Args:
            tensorList (inference_pb2.TensorList): snpe input tensor.

        Returns:
            dict[str, torch.tensor]: Inference results of snpe model.
        """
        resp = self.stub.Inference(tensorList)

        def get_shape(shape):
            if len(shape) == 4:
                if shape[0] == 1 and shape[
                        1] == 1 and shape[2] > 1 and shape[3] > 1:
                    # snpe NHWC layout works except for segmentation task
                    return (0, 1, 2, 3)
                return (0, 3, 1, 2)
            elif len(shape) == 3:
                return (0, 1, 2)
            elif len(shape) == 2:
                return (0, 1)
            return (0)

        result = dict()
        if resp.status == 0:
            for tensor in resp.data:
                ndarray = np.frombuffer(tensor.data, dtype=np.float32)
                shape = tuple(tensor.shape)
                data = torch.from_numpy(
                    ndarray.reshape(shape).copy()).to(device)
                data = data.permute(get_shape(data.shape))
                result[tensor.name] = data
        else:
            logger = get_root_logger()
            logger.error(f'snpe inference failed {resp.info}')

        return result
