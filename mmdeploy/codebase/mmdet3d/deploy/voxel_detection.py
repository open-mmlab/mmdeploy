from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.parallel import collate, scatter
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from torch.utils.data import DataLoader, Dataset

from mmdeploy.codebase.base import BaseTask
from mmdeploy.codebase.mmdet3d.deploy.mmdetection3d import MMDET3D_TASK
from mmdeploy.utils import Task, get_root_logger
from .voxel_detection_model import VoxelDetectionModel


class VoxelDetectionWrap(nn.Module):

    def __init__(self, model):
        super(VoxelDetectionWrap, self).__init__()
        self.model = model

    def forward(self, voxels, num_points, coors):
        result = self.model(
            voxel_input=[voxels, num_points, coors], img_metas=[0])
        return result[0], result[1], result[2]


@MMDET3D_TASK.register_module(Task.VOXEL_DETECTION.value)
class VoxelDetection(BaseTask):

    def __init__(self, model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                 device: str):
        super().__init__(model_cfg, deploy_cfg, device)

    def init_backend_model(self,
                           model_files: Sequence[str] = None,
                           **kwargs) -> torch.nn.Module:
        from .voxel_detection_model import build_voxel_detection_model
        model = build_voxel_detection_model(
            model_files, self.model_cfg, self.deploy_cfg, device=self.device)
        return model

    def init_pytorch_model(self,
                           model_checkpoint: Optional[str] = None,
                           cfg_options: Optional[Dict] = None,
                           is_onnx_export=False,
                           **kwargs) -> torch.nn.Module:
        from mmdet3d.apis import init_model
        model = init_model(self.model_cfg, model_checkpoint, self.device)
        if is_onnx_export:
            model = VoxelDetectionWrap(model)
        return model.eval()

    def create_input(self,
                     pcd: Union[str, np.ndarray],
                     img_shape=None,
                     **kwargs) -> Tuple[Dict, torch.Tensor]:
        data = VoxelDetection.read_data_from_pcd_file(pcd, self.model_cfg,
                                                      self.device)
        voxels, num_points, coors = VoxelDetectionModel.voxelize(
            data['points'][0], self.model_cfg)
        data['voxels'] = voxels
        data['num_points'] = num_points
        data['coors'] = coors
        return data, [voxels, num_points, coors]

    def visualize(self,
                  model: torch.nn.Module,
                  image: Union[str, np.ndarray],
                  result: list,
                  output_file: str,
                  window_name: str = '',
                  show_result: bool = False,
                  **kwargs):
        from mmdet3d.apis import show_result_meshlab
        data = VoxelDetection.read_data_from_pcd_file(image, self.model_cfg,
                                                      self.device)
        show_result_meshlab(
            data,
            result,
            output_file,
            0.3,
            show=show_result,
            snapshot=show_result,
            task='det')

    @staticmethod
    def read_data_from_pcd_file(pcd, model_cfg, device):
        if isinstance(pcd, (list, tuple)):
            pcd = pcd[0]
        test_pipeline = Compose(model_cfg.data.test.pipeline)
        box_type_3d, box_mode_3d = get_box_type(
            model_cfg.data.test.box_type_3d)
        data = dict(
            pts_filename=pcd,
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d,
            # for ScanNet demo we need axis_align_matrix
            ann_info=dict(axis_align_matrix=np.eye(4)),
            sweeps=[],
            # set timestamp = 0
            timestamp=[0],
            img_fields=[],
            bbox3d_fields=[],
            pts_mask_fields=[],
            pts_seg_fields=[],
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[])
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
        data['points'] = [point.data[0] for point in data['points']]
        if device != 'cpu':
            data = scatter(data, [device])[0]
        else:
            data['img_metas'] = data['img_metas'][0]
            data['points'] = data['points'][0]
        return data

    @staticmethod
    def run_inference(model, model_inputs: Dict[str, torch.Tensor]):
        result = model(
            return_loss=False,
            points=model_inputs['points'],
            img_metas=model_inputs['img_metas'])
        return [result]

    def get_tensor_from_input(self, input_data: Dict[str, Any],
                              **kwargs) -> torch.Tensor:
        pass

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
            print(dataset.evaluate(outputs, **eval_kwargs))

    def get_model_name(self) -> str:
        assert 'type' in self.model_cfg.model, 'model config contains no type'
        name = self.model_cfg.model.type.lower()
        return name

    def get_partition_cfg(partition_type: str, **kwargs) -> Dict:
        pass

    def get_postprocess(self) -> Dict:
        pass

    def get_preprocess(self) -> Dict:
        pass

    def single_gpu_test(self,
                        model: torch.nn.Module,
                        data_loader: DataLoader,
                        show: bool = False,
                        out_dir: Optional[str] = None,
                        **kwargs):
        model.eval()
        results = []
        dataset = data_loader.dataset

        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = model(data['points'][0].data,
                               data['img_metas'][0].data, False)
            results.extend(result)

            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()
        return results
