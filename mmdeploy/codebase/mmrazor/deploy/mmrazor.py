# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from mmengine import Config
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import Registry

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Codebase, Task

MMRAZOR_TASK = Registry('mmrazor_tasks')


@CODEBASE.register_module(Codebase.MMRAZOR.value)
class MMRazor(MMCodebase):
    """MMRazor codebase class."""
    task_registry = MMRAZOR_TASK

    @classmethod
    def register_deploy_modules(cls):
        """Register all rewriters for mmrazor."""
        pass

    @classmethod
    def register_all_modules(cls):
        """Register all related modules and rewriters for mmrazor."""
        from mmrazor.utils import register_all_modules
        register_all_modules(True)

    @classmethod
    def build_task_processor(cls, model_cfg: Config, deploy_cfg: Config,
                             device: str):
        """Build task processor for mmrazor.

        Now we use ModelCompress by default.
        """
        return ModelCompress(
            model_cfg=model_cfg, deploy_cfg=deploy_cfg, device=device)


@MMRAZOR_TASK.register_module(Task.ModelCompress.value)
class ModelCompress(BaseTask):
    """General model compress task for mmrazor.

    Args:
        model_cfg (Config): Original PyTorch model config file
        deploy_cfg (Config): Deployment config file or loaded Config
            object.
        device (str):  A string represents device type.
        experiment_name (str, optional): Name of current experiment.
            If not specified, timestamp will be used as
            ``experiment_name``. Defaults to ``None``.
    """

    def __init__(self,
                 model_cfg: Config,
                 deploy_cfg: Config,
                 device: str,
                 experiment_name: str = 'BaseTask'):

        super().__init__(model_cfg, deploy_cfg, device, experiment_name)
        self.origin_model_cfg = self.revert_model_cfg(model_cfg)
        self.base_task = build_task_processor(self.origin_model_cfg,
                                              deploy_cfg, device)

    def revert_model_cfg(self, model_cfg: Config):
        """Restore the original model config from the model config of the
        compressed model."""
        origin_model_cfg = copy.deepcopy(model_cfg)
        model = model_cfg['model']
        if 'architecture' in model:
            origin_model = model['architecture']
        elif 'algorithm' in model:
            origin_model = model['algorithm']['architecture']
        else:
            raise NotImplementedError()
        origin_model_cfg['model'] = origin_model
        if 'data_preprocessor' in origin_model:
            origin_model_cfg['data_preprocessor'] = origin_model[
                'data_preprocessor']
        return origin_model_cfg

    # abstract method

    def build_backend_model(self,
                            model_files=None,
                            data_preprocessor_updater=None,
                            **kwargs) -> torch.nn.Module:
        """Build backend model for using base task."""
        return self.base_task.build_backend_model(model_files,
                                                  data_preprocessor_updater,
                                                  **kwargs)

    def create_input(self,
                     imgs: Union[str, np.ndarray],
                     input_shape=None,
                     data_preprocessor: Optional[BaseDataPreprocessor] = None,
                     **kwargs) -> Tuple[Dict, torch.Tensor]:
        """Create input using base task."""
        return self.base_task.create_input(imgs, input_shape,
                                           data_preprocessor, **kwargs)

    def get_model_name(self, *args, **kwargs) -> str:
        """Get model name using base task."""
        return self.base_task.get_model_name(*args, **kwargs)

    def get_preprocess(self, *args, **kwargs) -> Dict:
        """Get data preprocess name using base task."""
        return self.base_task.get_preprocess(*args, **kwargs)

    def get_postprocess(self, *args, **kwargs) -> Dict:
        """Get data poseprocess name using base task."""
        return self.base_task.get_postprocess(*args, **kwargs)

    @staticmethod
    def get_partition_cfg(partition_type: str, **kwargs) -> Dict:
        """Get a certain partition config."""
        raise NotImplementedError()

    def build_pytorch_model(self,
                            model_checkpoint: Optional[str] = None,
                            cfg_options: Optional[Dict] = None,
                            **kwargs) -> torch.nn.Module:
        """Build PyTorch model for mmrazor and execute post process for
        mmdeploy."""
        model = super().build_pytorch_model(model_checkpoint, cfg_options,
                                            **kwargs)
        if hasattr(model, 'post_process_for_mmdeploy'):
            model.post_process_for_mmdeploy()

        return model
