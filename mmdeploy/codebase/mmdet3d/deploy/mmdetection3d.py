# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os.path as osp
from unittest import result
import numpy as np
import torch
import mmengine
from mmengine import Config
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import Registry
from mmengine.dataset import Compose, pseudo_collate
from .voxel_detection_model import VoxelDetectionModel
from mmdeploy.utils import Backend
# Copyright (c) OpenMMLab. All rights reserved.

import mmcv
import torch.nn as nn
# from mmcv.parallel import collate, scatter
from mmdet3d.structures import get_box_type


from torch.utils.data import DataLoader, Dataset

from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Codebase, Task, get_root_logger
from mmdeploy.utils.config_utils import get_input_shape, is_dynamic_shape

MMDET3D_TASK = Registry('mmdet3d_tasks')

@CODEBASE.register_module(Codebase.MMDET3D.value)
class MMDetection3d(MMCodebase):
    """MMDetection3d codebase class."""

    task_registry = MMDET3D_TASK


    # @staticmethod
    # def build_task_processor(model_cfg: mmengine.Config,
    #                          deploy_cfg: mmengine.Config,
    #                          device: str) -> BaseTask:
    #     """The interface to build the task processors of mmdet3d.

    #     Args:
    #         model_cfg (str | mmengine.Config): Model config file.
    #         deploy_cfg (str | mmengine.Config): Deployment config file.
    #         device (str): A string specifying device type.

    #     Returns:
    #         BaseTask: A task processor.
    #     """
    #     return MMDET3D_TASK.build(model_cfg, deploy_cfg, device)

    # @staticmethod
    # def build_dataset(dataset_cfg: Union[str, mmengine.Config], *args,
    #                   **kwargs) -> Dataset:
    #     """Build dataset for detection3d.

    #     Args:
    #         dataset_cfg (str | mmengine.Config): The input dataset config.

    #     Returns:
    #         Dataset: A PyTorch dataset.
    #     """
    #     from mmdet3d.datasets import build_dataset as build_dataset_mmdet3d

    #     from mmdeploy.utils import load_config
    #     dataset_cfg = load_config(dataset_cfg)[0]
    #     data = dataset_cfg.data

    #     dataset = build_dataset_mmdet3d(data.test)
    #     return dataset

    # @staticmethod
    # def build_dataloader(dataset: Dataset,
    #                      samples_per_gpu: int,
    #                      workers_per_gpu: int,
    #                      num_gpus: int = 1,
    #                      dist: bool = False,
    #                      shuffle: bool = False,
    #                      seed: Optional[int] = None,
    #                      runner_type: str = 'EpochBasedRunner',
    #                      persistent_workers: bool = True,
    #                      **kwargs) -> DataLoader:
    #     """Build dataloader for detection3d.

    #     Args:
    #         dataset (Dataset): Input dataset.
    #         samples_per_gpu (int): Number of training samples on each GPU, i.e.
    #             ,batch size of each GPU.
    #         workers_per_gpu (int): How many subprocesses to use for data
    #             loading for each GPU.
    #         num_gpus (int): Number of GPUs. Only used in non-distributed
    #             training.
    #         dist (bool): Distributed training/test or not.
    #             Defaults  to `False`.
    #         shuffle (bool): Whether to shuffle the data at every epoch.
    #             Defaults to `False`.
    #         seed (int): An integer set to be seed. Default is `None`.
    #         runner_type (str): Type of runner. Default: `EpochBasedRunner`.
    #         persistent_workers (bool): If True, the data loader will not
    #             shutdown the worker processes after a dataset has been consumed
    #             once. This allows to maintain the workers `Dataset` instances
    #             alive. This argument is only valid when PyTorch>=1.7.0.
    #             Default: False.
    #         kwargs: Any other keyword argument to be used to initialize
    #             DataLoader.

    #     Returns:
    #         DataLoader: A PyTorch dataloader.
    #     """
    #     from mmdet3d.datasets import \
    #         build_dataloader as build_dataloader_mmdet3d
    #     return build_dataloader_mmdet3d(
    #         dataset,
    #         samples_per_gpu,
    #         workers_per_gpu,
    #         num_gpus=num_gpus,
    #         dist=dist,
    #         shuffle=shuffle,
    #         seed=seed,
    #         runner_type=runner_type,
    #         persistent_workers=persistent_workers,
    #         **kwargs)

def _get_dataset_metainfo(model_cfg: Config):
    """Get metainfo of dataset.

    Args:
        model_cfg Config: Input model Config object.

    Returns:
        list[str]: A list of string specifying names of different class.
    """
    from mmdet3d.registry import DATASETS

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
            model_files, self.model_cfg, self.deploy_cfg, device=self.device,
            data_preprocessor=data_preprocessor)
        model = model.to(self.device)
        return model
    
    # def build_pytorch_model(self,
    #                         model_checkpoint: Optional[str] = None,
    #                         cfg_options: Optional[Dict] = None,
    #                         **kwargs) -> torch.nn.Module:
    #     """Initialize torch model.

    #     Args:
    #         model_checkpoint (str): The checkpoint file of torch model,
    #             defaults to `None`.
    #         cfg_options (dict): Optional config key-pair parameters.
    #     Returns:
    #         nn.Module: An initialized torch model generated by other OpenMMLab
    #             codebases.
    #     """
    #     from mmdet3d.apis import init_model
    #     device = self.device
    #     model = init_model(self.model_cfg, model_checkpoint, device)
    #     return model.eval()

    def create_input(
        self,
        pcd: str,
        input_shape: Sequence[int] = None,
        data_preprocessor: Optional[BaseDataPreprocessor] = None
    ) -> Tuple[Dict, torch.Tensor]:
        """Create input for detector.

        Args:
            pcd (str): Input pcd file path.
            input_shape (Sequence[int], optional): model input shape. Defaults to None.
            data_preprocessor (Optional[BaseDataPreprocessor], optional): model input preprocess. Defaults to None.

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
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)
        data_ = test_pipeline(data_)
        data.append(data_)
        
        collate_data = pseudo_collate(data)

        if data_preprocessor is not None:
            collate_data = data_preprocessor(collate_data, False)
            del collate_data['inputs']['voxels']['voxel_centers']
        return collate_data, collate_data['inputs']


    def visualize(self,
                  image: Union[str, np.ndarray],
                  model: torch.nn.Module,
                  result: list,
                  output_file: str,
                  window_name: str = '',
                  show_result: bool = False,
                  draw_gt: bool = False,
                  **kwargs):
        """_summary_

        Args:
            image (Union[str, np.ndarray]): _description_
            result (list): _description_
            output_file (str): _description_
            window_name (str, optional): _description_. Defaults to ''.
            show_result (bool, optional): _description_. Defaults to False.
            draw_gt (bool, optional): _description_. Defaults to False.
        """
        cfg = self.model_cfg
        # from mmdet3d.registry import VISUALIZERS
        # visualizer = VISUALIZERS.build(cfg.visualizer)
        visualizer = super().get_visualizer(window_name, output_file)
        visualizer.dataset_meta = _get_dataset_metainfo(cfg)
        
        # show the results
        _, data_input = self.create_input(pcd=image)
        
        if Backend(window_name) != Backend.PYTORCH:
            predictions = VoxelDetectionModel.postprocess(model_cfg=self.model_cfg, deploy_cfg=self.deploy_cfg, outs=result[0], metas=result[1])
        else:
            predictions = result
        # import pdb
        # pdb.set_trace()
        
        visualizer.add_datasample(
            window_name,
            dict(points=data_input['points'][0]),
            data_sample=predictions[0],
            draw_gt=False,
            show=True,
            wait_time=0,
            out_file=output_file,
            pred_score_thr=0.0,
            vis_task='det')

    @staticmethod
    def run_inference(model: nn.Module,
                      model_inputs: Dict[str, torch.Tensor]) -> List:
        """Run inference once for a object detection model of mmdet3d.

        Args:
            model (nn.Module): Input model.
            model_inputs (dict): A dict containing model inputs tensor and
                meta info.

        Returns:
            list: The predictions of model inference.
        """
        result = model(
            return_loss=False,
            points=model_inputs['points'],
            img_metas=model_inputs['img_metas'])
        return [result]

    @staticmethod
    def evaluate_outputs(model_cfg,
                         outputs: Sequence,
                         dataset: Dataset,
                         metrics: Optional[str] = None,
                         out: Optional[str] = None,
                         metric_options: Optional[dict] = None,
                         format_only: bool = False,
                         log_file: Optional[str] = None):
        if out:
            logger = get_root_logger()
            logger.info(f'\nwriting results to {out}')
            mmcv.dump(outputs, out)
        kwargs = {} if metric_options is None else metric_options
        if format_only:
            dataset.format_results(outputs, **kwargs)
        if metrics:
            eval_kwargs = model_cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=metrics, **kwargs))
            dataset.evaluate(outputs, **eval_kwargs)

    def get_model_name(self) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        raise NotImplementedError

    def get_tensor_from_input(self, input_data: Dict[str, Any],
                              **kwargs) -> torch.Tensor:
        """Get input tensor from input data.

        Args:
            input_data (dict): Input data containing meta info and image
                tensor.
        Returns:
            torch.Tensor: An image in `Tensor`.
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

    def get_postprocess(self) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Composed of the postprocess information.
        """
        raise NotImplementedError

    def get_preprocess(self) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        raise NotImplementedError

    def single_gpu_test(self,
                        model: nn.Module,
                        data_loader: DataLoader,
                        show: bool = False,
                        out_dir: Optional[str] = None,
                        **kwargs) -> List:
        """Run test with single gpu.

        Args:
            model (nn.Module): Input model from nn.Module.
            data_loader (DataLoader): PyTorch data loader.
            show (bool): Specifying whether to show plotted results. Defaults
                to `False`.
            out_dir (str): A directory to save results, defaults to `None`.

        Returns:
            list: The prediction results.
        """
        model.eval()
        results = []
        dataset = data_loader.dataset

        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = model(data['points'][0].data,
                               data['img_metas'][0].data, False)
            if show:
                # Visualize the results of MMDetection3D model
                # 'show_results' is MMdetection3D visualization API
                if out_dir is None:
                    model.module.show_result(
                        data,
                        result,
                        out_dir='',
                        file_name='',
                        show=show,
                        snapshot=False,
                        score_thr=0.3)
                else:
                    model.module.show_result(
                        data,
                        result,
                        out_dir=out_dir,
                        file_name=f'model_output{i}',
                        show=show,
                        snapshot=True,
                        score_thr=0.3)
            results.extend(result)

            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()
        return results
