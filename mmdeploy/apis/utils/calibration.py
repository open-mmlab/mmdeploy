# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..core import PIPELINE_MANAGER


@PIPELINE_MANAGER.register_pipeline()
def create_calib_input_data(calib_file: str,
                            model: torch.nn.Module,
                            dataloader: DataLoader,
                            get_tensor_func: Optional[Callable] = None,
                            inference_func: Optional[Callable] = None,
                            model_partition: bool = False,
                            context_info: Dict = dict(),
                            device: str = 'cpu') -> None:
    """Create calibration table.

    Examples:
        >>> from mmdeploy.apis.utils import create_calib_input_data
        >>> from mmdeploy.utils import get_calib_filename, load_config
        >>> deploy_cfg = 'configs/mmdet/detection/'
            'detection_tensorrt-int8_dynamic-320x320-1344x1344.py'
        >>> deploy_cfg = load_config(deploy_cfg)[0]
        >>> calib_file = get_calib_filename(deploy_cfg)
        >>> model_cfg = 'mmdetection/configs/fcos/'
            'fcos_r50_caffe_fpn_gn-head_1x_coco.py'
        >>> model_checkpoint = 'checkpoints/'
            'fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'
        >>> create_calib_input_data(calib_file, deploy_cfg,
            model_cfg, model_checkpoint, device='cuda:0')

    Args:
        calib_file (str): Input calibration file.
        deploy_cfg (str | mmcv.Config): Deployment config.
        model_cfg (str | mmcv.Config): The model config.
        model_checkpoint (str): PyTorch model checkpoint, defaults to `None`.
        dataset_cfg (str | mmcv.Config): Dataset config, defaults to `None`
        dataset_type (str): A string specifying dataset type, e.g.: 'test',
            'val', defaults to 'val'.
        device (str): Specifying the device to run on, defaults to 'cpu'.
    """
    import h5py
    import tqdm

    from mmdeploy.core import RewriterContext, reset_mark_function_count

    backend = 'default'

    with h5py.File(calib_file, mode='w') as file:
        calib_data_group = file.create_group('calib_data')

        if not model_partition:
            # create end2end group
            input_data_group = calib_data_group.create_group('end2end')
            input_group = input_data_group.create_group('input')
        for data_id, input_data in enumerate(tqdm.tqdm(dataloader)):

            if not model_partition:
                # save end2end data
                if get_tensor_func is not None:
                    input_tensor = get_tensor_func(input_data)
                else:
                    input_tensor = input_data
                input_ndarray = input_tensor.detach().cpu().numpy()
                input_group.create_dataset(
                    str(data_id),
                    shape=input_ndarray.shape,
                    compression='gzip',
                    compression_opts=4,
                    data=input_ndarray)
            else:
                context_info_ = deepcopy(context_info)
                if 'cfg' not in context_info:
                    context_info_['cfg'] = dict()
                context_info_['backend'] = backend
                context_info_['create_calib'] = True
                context_info_['calib_file'] = file
                context_info_['data_id'] = data_id

                with torch.no_grad(), RewriterContext(**context_info_):
                    reset_mark_function_count()
                    if inference_func is not None:
                        inference_func(model, input_data)
                    else:
                        model(input_data)

            file.flush()
