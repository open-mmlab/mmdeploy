# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Sequence

import torch


class BaseWrapper(torch.nn.Module, metaclass=ABCMeta):
    """Abstract base class for backend wrappers.

    Args:
        output_names (Sequence[str]): Names to model outputs in order, which is
        useful when converting the output dict to a ordered list or converting
        the output ordered list to a key-value dict.
    """

    def __init__(self, output_names: Sequence[str]):
        super().__init__()
        self._output_names = output_names

    @staticmethod
    def get_backend_file_count() -> int:
        """Return the count of backend file(s)

        Each backend has its own requirement on backend files (e.g., TensorRT
        requires 1 .engine file and ncnn requires 2 files (.param, .bin)). This
        interface allow developers to get the count of these required files.

        Returns:
            int: The count of required backend file(s).
        """
        return 1

    @abstractmethod
    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): Key-value pairs of model inputs.

        Returns:
            Dict[str, torch.Tensor]: Key-value pairs of model outputs.
        """
        pass

    @property
    def output_names(self):
        """Return the output names."""
        return self._output_names

    @output_names.setter
    def output_names(self, value):
        """Set the output names."""
        self._output_names = value

    def output_to_list(
            self, output_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Convert the output dict of forward() to a tensor list.

        Args:
            output_dict (Dict[str, torch.Tensor]): Key-value pairs of model
                outputs.

        Returns:
            List[torch.Tensor]: An output value list whose order is determined
                by the ouput_names list.
        """
        outputs = [output_dict[name] for name in self._output_names]
        return outputs
