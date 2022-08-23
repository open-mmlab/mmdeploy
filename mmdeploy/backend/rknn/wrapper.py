# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from rknn.api import RKNN

from mmdeploy.utils import Backend, get_root_logger
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper


@BACKEND_WRAPPER.register_module(Backend.RKNN.value)
class RKNNWrapper(BaseWrapper):
    """RKNN wrapper for inference.

    Args:
        model (str): Path of input RKNN model file.
        common_config (Dict): Config args for RKNN.
        output_names (Sequence[str]): Output names of the model.
        verbose (bool): Whether verbose during inference.

    Examples:
        >>> from mmdeploy.backend.rknn import RKNNWrapper
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
                 common_config: Dict = dict(target_platform=None),
                 output_names: Optional[Sequence[str]] = None,
                 verbose=True,
                 **kwargs):
        logger = get_root_logger()
        # Create RKNN object
        self.rknn = RKNN(verbose=verbose)
        self.rknn.load_rknn(model)
        ret = self.rknn.init_runtime(target=common_config['target_platform'])
        if ret != 0:
            logger.error('Init runtime environment failed!')
            exit(ret)
        super().__init__(output_names)

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Sequence[torch.Tensor]:
        """Run forward inference. Note that the shape of the input tensor is
        NxCxHxW while RKNN only accepts the numpy inputs of NxHxWxC. There is a
        permute operation outside RKNN inference.

        Args:
            inputs (Dict[str, torch.Tensor]): Input name and tensor pairs.

        Return:
            Sequence[torch.Tensor]: The output tensors.
        """
        rknn_out = self.__rknnnn_execute(
            [i.permute(0, 2, 3, 1).cpu().numpy() for i in inputs.values()])
        return [torch.from_numpy(out) for out in rknn_out]

    @TimeCounter.count_time(Backend.RKNN.value)
    def __rknnnn_execute(self, inputs: Sequence[np.array]):
        """Run inference with RKNN."""
        return self.rknn.inference(inputs)
