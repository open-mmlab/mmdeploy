# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import numpy as np
from rknn.api import RKNN
import torch

from mmdeploy.utils import Backend, get_root_logger
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper


@BACKEND_WRAPPER.register_module(Backend.RKNN.value)
class RKNNWrapper(BaseWrapper):
    """PPLNN wrapper for inference.

    Args:
        model (str): Path of input RKNN model file.
        target_platform (str): Device to put model.

    Examples:
        >>> from mmdeploy.backend.pplnn import PPLNNWrapper
        >>> import torch
        >>>
        >>> model = 'model.rknn'
        >>> model = RKNNWrapper(model)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self,
                 model: str,
                 common_config: Dict,
                 output_names: Optional[Sequence[str]] = None,
                 **kwargs):
        logger = get_root_logger()
        # Create RKNN object
        self.rknn = RKNN(verbose=True)
        # common_config.update(dict(mean_values=[0, 0, 0], std_values=[1, 1, 1]))
        # self.rknn.config(**common_config)
        self.rknn.load_rknn(model)
        ret = self.rknn.init_runtime(target=common_config['target_platform'])
        if ret != 0:
            logger.error('Init runtime environment failed!')
            exit(ret)
        output_names = ['pred_maps.0', 'pred_maps.1', 'pred_maps.2']
        super().__init__(output_names)

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): Input name and tensor pairs.

        Return:
            Dict[str, torch.Tensor]: The output name and tensor pairs.
        """
        # import cv2
        # img = cv2.imread(
        #     '/home/PJLAB/dongchunyu/dongchunyu/codes/rknn-toolkit2/examples/onnx/resnet50v2/dog_224x224.jpg'
        # )
        # # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (224, 224))
        rknn_out = self.__rknnnn_execute(
            [i.cpu().numpy() for i in inputs.values()])
        outputs = {}
        for i in range(len(self.output_names)):
            outputs[self.output_names[i]] = torch.from_numpy(rknn_out[i])
        return outputs

    @TimeCounter.count_time(Backend.RKNN.value)
    def __rknnnn_execute(self, inputs: Sequence[np.array]):
        """Run inference with PPLNN."""
        return self.rknn.inference(inputs)
