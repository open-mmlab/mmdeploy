from typing import Optional, Union

import h5py
import mmcv
import torch
import torch.multiprocessing as mp
from mmcv.parallel import MMDataParallel

from mmdeploy.core import (RewriterContext, patch_model,
                           reset_mark_function_count)
from .utils import (_inference, build_dataloader, build_dataset,
                    get_tensor_from_input, init_model)


def create_calib_table(calib_file: str,
                       deploy_cfg: Union[str, mmcv.Config],
                       model_cfg: Union[str, mmcv.Config],
                       model_checkpoint: Optional[str] = None,
                       dataset_cfg: Optional[Union[str, mmcv.Config]] = None,
                       dataset_type: str = 'val',
                       device: str = 'cuda:0',
                       ret_value: Optional[mp.Value] = None,
                       **kwargs) -> None:

    if ret_value is not None:
        ret_value.value = -1

    if dataset_cfg is None:
        dataset_cfg = model_cfg

    # load deploy_cfg if necessary
    if isinstance(deploy_cfg, str):
        deploy_cfg = mmcv.Config.fromfile(deploy_cfg)
    if not isinstance(deploy_cfg, mmcv.Config):
        raise TypeError('deploy_cfg must be a filename or Config object, '
                        f'but got {type(deploy_cfg)}')
    # load model_cfg if needed
    if isinstance(model_cfg, str):
        model_cfg = mmcv.Config.fromfile(model_cfg)
    if not isinstance(model_cfg, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(model_cfg)}')

    device_id = torch.device(device).index
    if device_id is None:
        device_id = 0

    # load dataset_cfg if needed
    if dataset_cfg is None:
        dataset_cfg = model_cfg
    elif isinstance(dataset_cfg, str):
        dataset_cfg = mmcv.Config.fromfile(dataset_cfg)
    if not isinstance(dataset_cfg, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(dataset_cfg)}')

    codebase = deploy_cfg.codebase
    apply_marks = deploy_cfg.get('apply_marks', False)
    backend = 'default'
    model = init_model(codebase, model_cfg, model_checkpoint, device=device)
    dataset = build_dataset(codebase, dataset_cfg, dataset_type)

    # patch model
    patched_model = patch_model(model, cfg=deploy_cfg, backend=backend)

    with h5py.File(calib_file, mode='w') as calib_file:
        calib_data_group = calib_file.create_group('calib_data')

        if not apply_marks:
            # create end2end group
            input_data_group = calib_data_group.create_group('end2end')
            input_group = input_data_group.create_group('input')
        dataloader = build_dataloader(
            codebase, dataset, 1, 1, dist=False, shuffle=False)
        patched_model = MMDataParallel(patched_model, device_ids=[device_id])
        prog_bar = mmcv.ProgressBar(len(dataset))
        for data_id, input_data in enumerate(dataloader):

            if not apply_marks:
                # save end2end data
                input_tensor = get_tensor_from_input(codebase, input_data)
                input_ndarray = input_tensor.detach().cpu().numpy()
                input_group.create_dataset(
                    str(data_id),
                    shape=input_ndarray.shape,
                    compression='gzip',
                    compression_opts=4,
                    data=input_ndarray)

            with torch.no_grad(), RewriterContext(
                    cfg=deploy_cfg,
                    backend=backend,
                    create_calib=True,
                    calib_file=calib_file,
                    data_id=data_id):
                reset_mark_function_count()
                _ = _inference(codebase, input_data, patched_model)
            calib_file.flush()

            prog_bar.update()

    if ret_value is not None:
        ret_value.value = 0
