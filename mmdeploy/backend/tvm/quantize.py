# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterable

import tvm
from tvm.runtime.ndarray import array


class IteratorDataset:
    """HDF5 dataset.

    Args:
        dataset (Iterable): Iterable dataset object.
        device (str): Device type, default to llvm.
    """

    def __init__(
        self,
        dataset: Iterable,
        device: str = 'llvm',
    ) -> None:

        self._dataset = dataset
        self._device = device

    def __call__(self):
        """Create dataset generator.

        Yields:
            Iterator[Any]: data in the dataset
        """
        for data in self._dataset:
            ret = dict()
            for name, data_np in data.items():
                # cast back to numpy
                try:
                    import torch
                    if isinstance(data_np, torch.Tensor):
                        data_np = data_np.detach().cpu().numpy()
                except Exception:
                    pass

                # tvm array
                ret[name] = array(data_np, tvm.device(self._device))

            yield ret
