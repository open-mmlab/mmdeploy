# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmengine.registry import Registry
from torch.utils.data import DataLoader

from mmdeploy.codebase.base import CODEBASE, MMCodebase
from mmdeploy.utils import Codebase

MMCLS_TASK = Registry('mmcls_tasks')


@CODEBASE.register_module(Codebase.MMCLS.value)
class MMClassification(MMCodebase):
    """mmclassification codebase class."""

    task_registry = MMCLS_TASK

    @staticmethod
    def single_gpu_test(model: torch.nn.Module,
                        data_loader: DataLoader,
                        show: bool = False,
                        out_dir: Optional[str] = None,
                        win_name: str = '',
                        **kwargs) -> List:
        """Run test with single gpu.

        Args:
            model (torch.nn.Module): Input model from nn.Module.
            data_loader (DataLoader): PyTorch data loader.
            show (bool): Specifying whether to show plotted results.
                Default: False.
            out_dir (str): A directory to save results, Default: None.
            win_name (str): The name of windows, Default: ''.

        Returns:
            list: The prediction results.
        """
        from mmcls.apis import single_gpu_test
        outputs = single_gpu_test(
            model, data_loader, show, out_dir, win_name=win_name, **kwargs)
        return outputs
