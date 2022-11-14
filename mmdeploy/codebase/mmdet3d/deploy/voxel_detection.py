# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Dict, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmdet3d.structures import get_box_type
from mmengine import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Codebase, Task

MMDET3D_TASK = Registry('mmdet3d_tasks')


@CODEBASE.register_module(Codebase.MMDET3D.value)
class MMDetection3d(MMCodebase):
    """MMDetection3d codebase class."""

    task_registry = MMDET3D_TASK

    @classmethod
    def register_deploy_modules(mmdet3d):
        import mmdeploy.codebase.mmdet3d.models  # noqa: F401

    @classmethod
    def register_all_modules(mmdet3d):
        from mmdet3d.utils.setup_env import register_all_modules

        mmdet3d.register_deploy_modules()
        register_all_modules(True)


def _get_dataset_metainfo(model_cfg: Config):
    """Get metainfo of dataset.

    Args:
        model_cfg Config: Input model Config object.

    Returns:
        list[str]: A list of string specifying names of different class.
    """

    for dataloader_name in [
            'test_dataloader', 'val_dataloader', 'train_dataloader'
    ]:
        if dataloader_name not in model_cfg:
            continue
        dataloader_cfg = model_cfg[dataloader_name]
        dataset_cfg = dataloader_cfg.dataset
        if 'metainfo' in dataset_cfg:
            return dataset_cfg.metainfo
    return None


@MMDET3D_TASK.register_module(Task.VOXEL_DETECTION.value)
class VoxelDetection(BaseTask):

    def __init__(self, model_cfg: mmengine.Config, deploy_cfg: mmengine.Config,
                 device: str):
        super().__init__(model_cfg, deploy_cfg, device)

    def build_backend_model(self,
                            model_files: Sequence[str] = None,
                            **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .voxel_detection_model import build_voxel_detection_model

        data_preprocessor = deepcopy(
            self.model_cfg.model.get('data_preprocessor', {}))
        data_preprocessor.setdefault('type', 'mmdet3D.Det3DDataPreprocessor')

        model = build_voxel_detection_model(
            model_files,
            self.model_cfg,
            self.deploy_cfg,
            device=self.device,
            data_preprocessor=data_preprocessor)
        model = model.to(self.device)
        return model

    def create_input(
        self,
        pcd: str,
        input_shape: Sequence[int] = None,
        data_preprocessor: Optional[BaseDataPreprocessor] = None
    ) -> Tuple[Dict, torch.Tensor]:
        """Create input for detector.

        Args:
            pcd (str): Input pcd file path.
            input_shape (Sequence[int], optional): model input shape.
                Defaults to None.
            data_preprocessor (Optional[BaseDataPreprocessor], optional):
                model input preprocess. Defaults to None.

        Returns:
            tuple: (data, input), meta information for the input pcd
                and model input.
        """

        cfg = self.model_cfg
        test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
        test_pipeline = Compose(test_pipeline)
        box_type_3d, box_mode_3d = \
            get_box_type(cfg.test_dataloader.dataset.box_type_3d)

        data = []
        data_ = dict(
            lidar_points=dict(lidar_path=pcd),
            timestamp=1,
            # for ScanNet demo we need axis_align_matrix
            axis_align_matrix=np.eye(4),
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d)
        data_ = test_pipeline(data_)
        data.append(data_)

        collate_data = pseudo_collate(data)
        data[0]['inputs']['points'] = data[0]['inputs']['points'].to(
            self.device)

        if data_preprocessor is not None:
            collate_data = data_preprocessor(collate_data, False)
            voxels = collate_data['inputs']['voxels']
            inputs = [voxels['voxels'], voxels['num_points'], voxels['coors']]
        else:
            inputs = collate_data['inputs']
        return collate_data, inputs

    def visualize(self,
                  image: Union[str, np.ndarray],
                  model: torch.nn.Module,
                  result: list,
                  output_file: str,
                  window_name: str = '',
                  show_result: bool = False,
                  draw_gt: bool = False,
                  **kwargs):
        """visualize backend output.

        Args:
            image (Union[str, np.ndarray]): pcd file path
            model (torch.nn.Module): input pytorch model
            result (list): output bbox, score and type
            output_file (str): the directory to save output
            window_name (str, optional): display window name
            show_result (bool, optional): show result or not.
                Defaults to False.
            draw_gt (bool, optional): show gt or not. Defaults to False.
        """
        cfg = self.model_cfg
        visualizer = super().get_visualizer(window_name, output_file)
        visualizer.dataset_meta = _get_dataset_metainfo(cfg)

        # show the results
        collate_data, _ = self.create_input(pcd=image)

        visualizer.add_datasample(
            window_name,
            dict(points=collate_data['inputs']['points'][0]),
            data_sample=result,
            draw_gt=draw_gt,
            show=show_result,
            wait_time=0,
            out_file=output_file,
            pred_score_thr=0.0,
            vis_task='lidar_det')

    def get_model_name(self, *args, **kwargs) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        raise NotImplementedError

    def get_partition_cfg(partition_type: str, **kwargs) -> Dict:
        """Get a certain partition config for mmdet.

        Args:
            partition_type (str): A string specifying partition type.

        Returns:
            dict: A dictionary of partition config.
        """
        raise NotImplementedError

    def get_postprocess(self, *args, **kwargs) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Composed of the postprocess information.
        """
        raise NotImplementedError

    def get_preprocess(self, *args, **kwargs) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        raise NotImplementedError
