from typing import Any, Dict, Optional, Sequence, Union

import mmcv
import numpy as np
import torch
from torch.utils.data import Dataset

from mmdeploy.utils import Backend, Codebase, Task, get_codebase, load_config


def init_pytorch_model(codebase: Codebase,
                       model_cfg: Union[str, mmcv.Config],
                       model_checkpoint: Optional[str] = None,
                       device: str = 'cuda:0',
                       cfg_options: Optional[Dict] = None):
    """Initialize torch model.

    Args:
        codebase (Codebase): Specifying codebase type.
        model_cfg (str | mmcv.Config): Model config file or Config object.
        model_checkpoint (str): The checkpoint file of torch model, defaults
            to `None`.
        device (str): A string specifying device type, defaults to 'cuda:0'.
        cfg_options (dict): Optional config key-pair parameters.

    Returns:
        nn.Module: An initialized torch model.
    """
    if codebase == Codebase.MMCLS:
        from mmcls.apis import init_model
        model = init_model(model_cfg, model_checkpoint, device, cfg_options)

    elif codebase == Codebase.MMDET:
        from mmdet.apis import init_detector
        model = init_detector(model_cfg, model_checkpoint, device, cfg_options)

    elif codebase == Codebase.MMSEG:
        from mmseg.apis import init_segmentor
        from mmdeploy.mmseg.export import convert_syncbatchnorm
        model = init_segmentor(model_cfg, model_checkpoint, device)
        model = convert_syncbatchnorm(model)

    elif codebase == Codebase.MMOCR:
        from mmocr.apis import init_detector
        model = init_detector(model_cfg, model_checkpoint, device, cfg_options)

    elif codebase == Codebase.MMEDIT:
        from mmedit.apis import init_model
        model = init_model(model_cfg, model_checkpoint, device)
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase.value}')

    return model.eval()


def create_input(codebase: Codebase,
                 task: Task,
                 model_cfg: Union[str, mmcv.Config],
                 imgs: Any,
                 input_shape: Sequence[int] = None,
                 device: str = 'cuda:0',
                 **kwargs):
    """Create input for model.

    Args:
        codebase (Codebase): Specifying codebase type.
        task (Task): Specifying task type.
        model_cfg (str | mmcv.Config): model config file or loaded Config
            object.
        imgs (str | np.ndarray): Input image(s).
        input_shape (list[int]): Input shape of image in (width, height)
            format, defaults to `None`.
        device (str): A string specifying device type, defaults to 'cuda:0'.

    Returns:
        tuple: (data, img), meta information for the input image and input
            image tensor.
    """
    model_cfg = load_config(model_cfg)[0]

    cfg = model_cfg.copy()
    if codebase == Codebase.MMCLS:
        from mmdeploy.mmcls.export import create_input
        return create_input(task, cfg, imgs, input_shape, device, **kwargs)

    elif codebase == Codebase.MMDET:
        from mmdeploy.mmdet.export import create_input
        return create_input(task, cfg, imgs, input_shape, device, **kwargs)

    elif codebase == Codebase.MMOCR:
        from mmdeploy.mmocr.export import create_input
        return create_input(task, cfg, imgs, input_shape, device, **kwargs)

    elif codebase == Codebase.MMSEG:
        from mmdeploy.mmseg.export import create_input
        return create_input(task, cfg, imgs, input_shape, device, **kwargs)

    elif codebase == Codebase.MMEDIT:
        from mmdeploy.mmedit.export import create_input
        return create_input(task, cfg, imgs, input_shape, device, **kwargs)

    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase.value}')


def init_backend_model(model_files: Sequence[str],
                       model_cfg: Union[str, mmcv.Config],
                       deploy_cfg: Union[str, mmcv.Config],
                       device_id: int = 0,
                       **kwargs):
    """Initialize backend model.

    Args:
        model_files (list[str]): Input model files.
        model_cfg (str | mmcv.Config): Model config file or
            loaded Config object.
        deploy_cfg (str | mmcv.Config): Deployment config file or
            loaded Config object.
        device_id (int): An integer specifying device index.

    Returns:
        nn.Module: An initialized model.
    """
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    codebase = get_codebase(deploy_cfg)

    if codebase == Codebase.MMCLS:
        from mmdeploy.mmcls.apis import build_classifier
        return build_classifier(
            model_files, model_cfg, deploy_cfg, device_id=device_id)

    elif codebase == Codebase.MMDET:
        from mmdeploy.mmdet.apis import build_detector
        return build_detector(
            model_files, model_cfg, deploy_cfg, device_id=device_id)

    elif codebase == Codebase.MMSEG:
        from mmdeploy.mmseg.apis import build_segmentor
        return build_segmentor(
            model_files, model_cfg, deploy_cfg, device_id=device_id)

    elif codebase == Codebase.MMOCR:
        from mmdeploy.mmocr.apis import build_ocr_processor
        return build_ocr_processor(
            model_files, model_cfg, deploy_cfg, device_id=device_id)

    elif codebase == Codebase.MMEDIT:
        from mmdeploy.mmedit.apis import build_editing_processor
        return build_editing_processor(model_files, model_cfg, deploy_cfg,
                                       device_id)

    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase.value}')


def run_inference(codebase: Codebase, model_inputs: dict,
                  model: torch.nn.Module):
    """Run once inference for a model of nn.Module.

    Args:
        codebase (Codebase): Specifying codebase type.
        model_inputs (dict): A dict containing model inputs tensor and
            meta info.
        model (nn.Module): Input model.

    Returns:
        list: The predictions of model inference.
    """
    if codebase == Codebase.MMCLS:
        return model(**model_inputs, return_loss=False)[0]
    elif codebase == Codebase.MMDET:
        return model(**model_inputs, return_loss=False, rescale=True)[0]
    elif codebase == Codebase.MMSEG:
        return model(**model_inputs, return_loss=False)
    elif codebase == Codebase.MMOCR:
        return model(**model_inputs, return_loss=False, rescale=True)[0]
    elif codebase == Codebase.MMEDIT:
        result = model(model_inputs['lq'])[0]
        # TODO: (For mmedit codebase)
        # The data type of pytorch backend is not consistent
        if not isinstance(result, np.ndarray):
            result = result.detach().cpu().numpy()
        return result
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase.value}')


def visualize(codebase: Codebase,
              image: Union[str, np.ndarray],
              result: list,
              model: torch.nn.Module,
              output_file: str,
              backend: Backend,
              show_result: bool = False):
    """Visualize predictions of a model.

    Args:
        codebase (Codebase): Specifying codebase type.
        image (str | np.ndarray): Input image to draw predictions on.
        result (list): A list of predictions.
        model (nn.Module): Input model.
        output_file (str): Output file to save drawn image.
        backend (Backend): Specifying backend type.
        show_result (bool): Whether to show result in windows, defaults
            to `False`.
    """
    show_img = mmcv.imread(image) if isinstance(image, str) else image
    output_file = None if show_result else output_file

    if codebase == Codebase.MMCLS:
        from mmdeploy.mmcls.apis import show_result as show_result_mmcls
        show_result_mmcls(model, show_img, result, output_file, backend,
                          show_result)
    elif codebase == Codebase.MMDET:
        from mmdeploy.mmdet.apis import show_result as show_result_mmdet
        show_result_mmdet(model, show_img, result, output_file, backend,
                          show_result)
    elif codebase == Codebase.MMSEG:
        from mmdeploy.mmseg.apis import show_result as show_result_mmseg
        show_result_mmseg(model, show_img, result, output_file, backend,
                          show_result)
    elif codebase == Codebase.MMOCR:
        from mmdeploy.mmocr.apis import show_result as show_result_mmocr
        show_result_mmocr(model, show_img, result, output_file, backend,
                          show_result)
    elif codebase == Codebase.MMEDIT:
        from mmdeploy.mmedit.apis import show_result as show_result_mmedit
        show_result_mmedit(result, output_file, backend, show_result)


def get_partition_cfg(codebase: Codebase, partition_type: str):
    """Get a certain partition config.

    Notes:
        Currently only support mmdet codebase.

    Args:
        codebase (Codebase): Specifying codebase type.
        partition_type (str): A string specifying partition type.

    Returns:
        dict: A dictionary of partition config.
    """
    if codebase == Codebase.MMDET:
        from mmdeploy.mmdet.export import get_partition_cfg \
            as get_partition_cfg_mmdet
        return get_partition_cfg_mmdet(partition_type)
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase.value}')


def build_dataset(codebase: Codebase,
                  dataset_cfg: Union[str, mmcv.Config],
                  dataset_type: str = 'val',
                  **kwargs):
    """Build dataset for different codebase.

    Args:
        codebase (Codebase): Specifying codebase type.
        dataset_cfg (str | mmcv.Config): Dataset config file or Config object.
        dataset_type (str): Specifying dataset type, e.g.: 'train', 'test',
            'val', defaults to 'val'.

    Returns:
        Dataset: The built dataset.
    """
    if codebase == Codebase.MMCLS:
        from mmdeploy.mmcls.export import build_dataset \
            as build_dataset_mmcls
        return build_dataset_mmcls(dataset_cfg, dataset_type, **kwargs)
    elif codebase == Codebase.MMDET:
        from mmdeploy.mmdet.export import build_dataset \
            as build_dataset_mmdet
        return build_dataset_mmdet(dataset_cfg, dataset_type, **kwargs)
    elif codebase == Codebase.MMSEG:
        from mmdeploy.mmseg.export import build_dataset as build_dataset_mmseg
        return build_dataset_mmseg(dataset_cfg, dataset_type, **kwargs)
    elif codebase == Codebase.MMEDIT:
        from mmdeploy.mmedit.export import build_dataset \
            as build_dataset_mmedit
        return build_dataset_mmedit(dataset_cfg, **kwargs)
    elif codebase == Codebase.MMOCR:
        from mmdeploy.mmocr.export import build_dataset as build_dataset_mmocr
        return build_dataset_mmocr(dataset_cfg, dataset_type, **kwargs)
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase.value}')


def build_dataloader(codebase: Codebase, dataset: Dataset,
                     samples_per_gpu: int, workers_per_gpu: int, **kwargs):
    """Build PyTorch dataloader.

    Args:
        codebase (Codebase): Specifying codebase type.
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    if codebase == Codebase.MMCLS:
        from mmdeploy.mmcls.export import build_dataloader \
            as build_dataloader_mmcls
        return build_dataloader_mmcls(dataset, samples_per_gpu,
                                      workers_per_gpu, **kwargs)
    elif codebase == Codebase.MMDET:
        from mmdeploy.mmdet.export import build_dataloader \
            as build_dataloader_mmdet
        return build_dataloader_mmdet(dataset, samples_per_gpu,
                                      workers_per_gpu, **kwargs)
    elif codebase == Codebase.MMSEG:
        from mmdeploy.mmseg.export import build_dataloader \
            as build_dataloader_mmseg
        return build_dataloader_mmseg(dataset, samples_per_gpu,
                                      workers_per_gpu, **kwargs)
    elif codebase == Codebase.MMEDIT:
        from mmdeploy.mmedit.export import build_dataloader \
            as build_dataloader_mmedit
        return build_dataloader_mmedit(dataset, samples_per_gpu,
                                       workers_per_gpu, **kwargs)
    elif codebase == Codebase.MMOCR:
        from mmdeploy.mmocr.export import build_dataloader \
            as build_dataloader_mmocr
        return build_dataloader_mmocr(dataset, samples_per_gpu,
                                      workers_per_gpu, **kwargs)
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase.value}')


def get_tensor_from_input(codebase: Codebase, input_data: tuple):
    """Get input tensor from input data.

    Args:
        codebase (Codebase): Specifying codebase type.
        input_data (tuple): Input data containing meta info and image tensor.

    Returns:
        torch.Tensor: Input tensor of image.
    """
    if codebase == Codebase.MMCLS:
        from mmdeploy.mmcls.export import get_tensor_from_input \
            as get_tensor_from_input_mmcls
        return get_tensor_from_input_mmcls(input_data)
    elif codebase == Codebase.MMDET:
        from mmdeploy.mmdet.export import get_tensor_from_input \
            as get_tensor_from_input_mmdet
        return get_tensor_from_input_mmdet(input_data)
    elif codebase == Codebase.MMSEG:
        from mmdeploy.mmseg.export import get_tensor_from_input \
            as get_tensor_from_input_mmseg
        return get_tensor_from_input_mmseg(input_data)
    elif codebase == Codebase.MMOCR:
        from mmdeploy.mmocr.export import get_tensor_from_input \
            as get_tensor_from_input_mmocr
        return get_tensor_from_input_mmocr(input_data)
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase.value}')
