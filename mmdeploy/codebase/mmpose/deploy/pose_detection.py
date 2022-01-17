# Copyright (c) OpenMMLab. All rights reserved.
import logging
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from torch.utils.data import Dataset

from mmdeploy.codebase.base import BaseTask
from mmdeploy.codebase.mmpose.deploy.mmpose import MMPOSE_TASK
from mmdeploy.utils import Task, get_input_shape, load_config


def process_model_config(model_cfg: mmcv.Config,
                         imgs: Union[Sequence[str], Sequence[np.ndarray]],
                         input_shape: Optional[Sequence[int]] = None):
    cfg = model_cfg.copy()
    return cfg


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


# @MMPOSE_TASK.register_module(Task.SUPER_RESOLUTION.value)
class PoseDetection(BaseTask):

    def __init__(self, model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                 device: str):
        super().__init__(model_cfg, deploy_cfg, device)

    def init_backend_model(self,
                           model_files: Sequence[str] = None,
                           **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files. Default is None.

        Returns:
            nn.Module: An initialized backend model.
        """
        # from pose_detection_model import build_pose_detection_model

    def init_pytorch_model(self,
                           model_checkpoint: Optional[str] = None,
                           **kwargs) -> torch.nn.Module:
        from mmpose.apis import init_pose_model
        model = init_pose_model(self.model_cfg, model_checkpoint, self.device)
        model = _convert_batchnorm(model)
        if hasattr(model, 'forward_dummy'):
            model.forward = model.forward_dummy
        else:
            raise NotImplementedError(
                'Please implement the forward method for exporting.')
        return model

    def create_input(self,
                     imgs: Union[str, np.ndarray],
                     input_shape: Sequence[int] = None,
                     **kwargs) -> Tuple[Dict, torch.Tensor]:
        from mmpose.datasets.pipelines import Compose

        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        cfg = process_model_config(self.model_cfg, imgs, input_shape)
        test_pipeline = Compose(cfg.test_pipeline)
        data_list = []
        for img in imgs:
            if isinstance(img, np.ndarray):
                # directly add img
                data = dict(image_file=img)
            else:
                # add information into dict
                data = dict(image_file=img)

            data = test_pipeline(data)
            data_list.append(data)

    def visualize(self,
                  model: torch.nn.Module,
                  image: Union[str, np.ndarray],
                  result: list,
                  output_file: str,
                  window_name: str,
                  show_result: bool = False,
                  score_thr: float = 0.3):
        pass

    def evaluate_outputs(model_cfg,
                         outputs: Sequence,
                         dataset: Dataset,
                         metrics: Optional[str] = None,
                         out: Optional[str] = None,
                         metric_options: Optional[dict] = None,
                         format_only: bool = False,
                         **kwargs):
        pass

    def get_model_name(self) -> str:
        pass

    def get_partition_cfg(partition_type: str, **kwargs) -> Dict:
        pass

    def get_preprocess(self) -> Dict:
        pass

    def get_postprocess(self) -> Dict:
        pass

    def get_tensor_from_input(self, input_data: Dict[str, Any],
                              **kwargs) -> torch.Tensor:
        pass

    def run_inference(model, model_inputs: Dict[str, torch.Tensor]):
        pass


if __name__ == '__main__':
    model_cfg = '/home/PJLAB/shenkun/workspace/mmpose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/hrnet_w32_animalpose_256x256.py'
    deploy_cfg = '/home/PJLAB/shenkun/workspace/mmdeploy/configs/mmpose/posedetection_static.py'
    checkpoint = '/home/PJLAB/shenkun/workspace/mmpose/checkpoints/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth'
    device = 'cuda:0'
    task = PoseDetection(model_cfg, deploy_cfg, device)
    img = torch.rand(1, 3, 256, 256)
    from mmpose.datasets.pipelines import Compose
    cfg = mmcv.Config.fromfile(model_cfg)
    img = task.create_input(img)
    test_pipeline = Compose(cfg.test_pipeline)
    print(test_pipeline(img))
