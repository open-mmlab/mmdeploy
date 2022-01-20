from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from mmdeploy.codebase.base import BaseTask
from mmdeploy.codebase.mmdet3d.deploy.mmdetection3d import MMDET3D_TASK
from mmdeploy.utils import Task


def voxelize(points, model_cfg):
    from mmdet3d.ops import Voxelization
    voxel_layer = model_cfg.model['voxel_layer']
    voxel_layer = Voxelization(**voxel_layer)
    voxels, coors, num_points = [], [], []
    for res in points:
        res_voxels, res_coors, res_num_points = voxel_layer(res)
        voxels.append(res_voxels)
        coors.append(res_coors)
        num_points.append(res_num_points)
    voxels = torch.cat(voxels, dim=0)
    num_points = torch.cat(num_points, dim=0)
    coors_batch = []
    for i, coor in enumerate(coors):
        coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
        coors_batch.append(coor_pad)
    coors_batch = torch.cat(coors_batch, dim=0)
    return voxels, num_points, coors_batch


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
                           **kwargs) -> torch.nn.Module:
        from mmdet3d.apis import init_model
        model = init_model(self.model_cfg, model_checkpoint, self.device)
        model = VoxelDetectionWrap(model)
        return model.eval()

    def create_input(self,
                     pcds: Union[str, np.ndarray],
                     img_shape=None,
                     **kwargs) -> Tuple[Dict, torch.Tensor]:

        from mmdet3d.datasets.pipelines import Compose
        from mmcv.parallel import collate, scatter
        from mmdet3d.core.bbox import get_box_type
        if not isinstance(pcds, (list, tuple)):
            pcds = [pcds]
        cfg = self.model_cfg
        test_pipeline = Compose(cfg.data.test.pipeline)
        box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
        data_list = []
        for pcd in pcds:
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
            data_list.append(data)

        result = collate(data_list, samples_per_gpu=len(pcds))
        result['img_metas'] = [
            img_metas.data[0] for img_metas in result['img_metas']
        ]
        result['points'] = [point.data[0] for point in result['points']]
        if self.device != 'cpu':
            result = scatter(result, [self.device])[0]
        else:
            result['img_metas'] = result['img_metas'][0]
            result['points'] = result['points'][0]
        voxels_batch, num_points_batch, coors_batch = [], [], []
        for point in result['points'][0]:
            voxels, num_points, coors = voxelize([point], self.model_cfg)
            voxels_batch.append(voxels)
            num_points_batch.append(num_points)
            coors_batch.append(coors)
        result['voxels'] = voxels_batch
        result['num_points'] = num_points_batch
        result['coors'] = coors_batch
        return result, [voxels_batch[0], num_points_batch[0], coors_batch[0]]

    def visualize(self,
                  model: torch.nn.Module,
                  image: Union[str, np.ndarray],
                  result: list,
                  output_file: str,
                  window_name: str = '',
                  show_result: bool = False,
                  **kwargs):
        print(result)

    @staticmethod
    def run_inference(model, model_inputs: Dict[str, torch.Tensor]):
        batch_size = len(model_inputs['voxels'])
        result = []
        for i in range(batch_size):
            voxels = model_inputs['voxels'][i]
            num_points = model_inputs['num_points'][i]
            coors = model_inputs['coors'][i]
            result.append(model(voxels, num_points, coors))
        return result

    def get_tensor_from_input(self, input_data: Dict[str, Any],
                              **kwargs) -> torch.Tensor:
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
        assert 'type' in self.model_cfg.model, 'model config contains no type'
        name = self.model_cfg.model.type.lower()
        return name

    def get_partition_cfg(partition_type: str, **kwargs) -> Dict:
        pass

    def get_postprocess(self) -> Dict:
        pass

    def get_preprocess(self) -> Dict:
        pass
