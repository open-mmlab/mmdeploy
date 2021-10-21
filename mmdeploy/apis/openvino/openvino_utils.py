import os.path as osp
from typing import Dict, List

import mmcv
import numpy as np
import torch

from mmdeploy.utils.timer import TimeCounter


def get_input_shape_from_cfg(config: mmcv.Config) -> List[int]:
    """Get the input shape from the model config for OpenVINO Model Optimizer.

    Args:
        config (mmcv.Config): Model config.
    Returns:
        List[int]: The input shape in [1, 3, H, W] format from config
            or [1, 3, 800, 1344].
    """
    shape = []
    test_pipeline = config.get('test_pipeline', None)
    if test_pipeline is not None:
        img_scale = test_pipeline[1]['img_scale']
        shape = [1, 3, img_scale[1], img_scale[0]]
    else:
        shape = [1, 3, 800, 1344]
    return shape


class OpenVINOWrapper(torch.nn.Module):
    """OpenVINO wrapper for inference in CPU.

    Args:
        ir_model_file (str): Input OpenVINO IR model file.

    Examples:
        >>> from mmdeploy.apis.openvino import OpenVINOWrapper
        >>> import torch
        >>>
        >>> ir_model_file = 'model.xml'
        >>> model = OpenVINOWrapper(ir_model_file)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224, device='cpu'))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self, ir_model_file: str):
        super(OpenVINOWrapper, self).__init__()
        from openvino.inference_engine import IECore
        self.ie = IECore()
        bin_path = osp.splitext(ir_model_file)[0] + '.bin'
        self.net = self.ie.read_network(ir_model_file, bin_path)
        for input in self.net.input_info.values():
            batch_size = input.input_data.shape[0]
            assert batch_size == 1, 'Only batch 1 is supported.'
        self.device = 'cpu'
        self.sess = self.ie.load_network(
            network=self.net, device_name=self.device.upper(), num_requests=1)

    def __update_device(
            self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Updates the device type to 'self.device' (cpu) for the input
        tensors.

        Args:
            inputs (Dict[str, torch.Tensor]): The input name and tensor pairs.

        Returns:
            Dict[str, torch.Tensor]: The output name and tensor pairs
                with updated device type.
        """
        updated_inputs = {
            name: data.to(torch.device(self.device))
            for name, data in inputs.items()
        }
        return updated_inputs

    def __reshape(self, inputs: Dict[str, torch.Tensor]):
        """Reshape the model for the shape of the input data.

        Args:
            inputs (Dict[str, torch.Tensor]): The input name and tensor pairs.
        """
        input_shapes = {name: data.shape for name, data in inputs.items()}
        reshape_needed = False
        for input_name, input_shape in input_shapes.items():
            blob_shape = self.net.input_info[input_name].input_data.shape
            if not np.array_equal(input_shape, blob_shape):
                reshape_needed = True
                break
        if reshape_needed:
            self.net.reshape(input_shapes)
            self.sess = self.ie.load_network(
                network=self.net,
                device_name=self.device.upper(),
                num_requests=1)

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): The input name and tensor pairs.

        Returns:
            Dict[str, torch.Tensor]: The output name and tensor pairs.
        """
        inputs = self.__update_device(inputs)
        self.__reshape(inputs)
        outputs = self.openvino_execute(inputs)
        return outputs

    @TimeCounter.count_time()
    def openvino_execute(
            self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run inference with OpenVINO IE.

        Args:
            inputs (Dict[str, torch.Tensor]): The input name and tensor pairs.

        Returns:
            Dict[str, torch.Tensor]: The output name and tensor pairs.
        """
        outputs = self.sess.infer(inputs)
        return outputs
