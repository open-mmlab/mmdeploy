# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import re
from typing import Dict, List, Tuple, Union

import mmengine

from mmdeploy.apis import build_task_processor
from mmdeploy.utils import (Backend, Task, get_backend, get_codebase,
                            get_ir_config, get_partition_config, get_precision,
                            get_root_logger, get_task_type, is_dynamic_batch,
                            load_config)
from mmdeploy.utils.config_utils import get_backend_config
from mmdeploy.utils.constants import SDK_TASK_MAP as task_map


def get_mmdeploy_version() -> str:
    """Return the version of MMDeploy."""
    import mmdeploy
    version = mmdeploy.__version__
    return version


def get_task(deploy_cfg: mmengine.Config) -> Dict:
    """Get the task info for mmdeploy.json.

    The task info is composed of
    task_name, the codebase name and the codebase version.
    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.
    Return:
        dict: The task info.
    """
    task_name = get_task_type(deploy_cfg).value
    codebase_name = get_codebase(deploy_cfg).value
    try:
        codebase = importlib.import_module(codebase_name)
    except ModuleNotFoundError:
        logger = get_root_logger()
        logger.warning(f'can not import the module: {codebase_name}')
    codebase_version = codebase.__version__
    return dict(
        task=task_name, codebase=codebase_name, version=codebase_version)


def get_model_name_customs(deploy_cfg: mmengine.Config,
                           model_cfg: mmengine.Config, work_dir: str,
                           device: str) -> Tuple:
    """Get the model name and dump custom file.

    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.
        model_cfg (mmengine.Config): The model config dict.
        work_dir (str): Work dir to save json files.
        device (str): The device passed in.

    Return:
        tuple(): Composed of the model name and the custom info.
    """
    task = get_task_type(deploy_cfg)
    task_processor = build_task_processor(
        model_cfg=model_cfg, deploy_cfg=deploy_cfg, device=device)
    name = task_processor.get_model_name()
    customs = []
    if task == Task.TEXT_RECOGNITION:
        customs.append('dict_file.txt')
    return name, customs


def get_models(deploy_cfg: Union[str, mmengine.Config],
               model_cfg: Union[str, mmengine.Config], work_dir: str,
               device: str) -> List:
    """Get the output model information for deploy.json.

    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.
        model_cfg (mmengine.Config): The model config dict.
        work_dir (str): Work dir to save json files.
        device (str): The device passed in.

    Return:
        list[dict]: The list contains dicts composed of the model name, net,
            weight, backend, precision batchsize and dynamic_shape.
    """
    name, _ = get_model_name_customs(deploy_cfg, model_cfg, work_dir, device)
    precision = 'FP32'
    ir_name = get_ir_config(deploy_cfg)['save_file']
    if get_partition_config(deploy_cfg) is not None:
        ir_name = get_partition_config(
            deploy_cfg)['partition_cfg'][0]['save_file']
    weights = ''
    backend = get_backend(deploy_cfg=deploy_cfg).value

    backend_net = dict(
        tensorrt=lambda file: re.sub(r'\.[a-z]+', '.engine', file),
        openvino=lambda file: re.sub(r'\.[a-z]+', '.xml', file),
        ncnn=lambda file: re.sub(r'\.[a-z]+', '.param', file),
        ascend=lambda file: re.sub(r'\.[a-z]+', '.om', file),
        rknn=lambda file: re.sub(r'\.[a-z]+', '.rknn', file),
        coreml=lambda file: re.sub(r'\.[a-z]+', '.mlpackage', file),
        snpe=lambda file: re.sub(r'\.[a-z]+', '.dlc', file))
    backend_weights = dict(
        pplnn=lambda file: re.sub(r'\.[a-z]+', '.json', file),
        openvino=lambda file: re.sub(r'\.[a-z]+', '.bin', file),
        ncnn=lambda file: re.sub(r'\.[a-z]+', '.bin', file))
    if backend != Backend.TVM.value:
        net = backend_net.get(backend, lambda x: x)(ir_name)
        weights = backend_weights.get(backend, lambda x: weights)(ir_name)
    else:
        # TODO: add this to backend manager
        import os.path as osp

        from mmdeploy.backend.tvm import get_library_ext

        def _replace_suffix(file_name: str, dst_suffix: str) -> str:
            return re.sub(r'\.[a-z]+', dst_suffix, file_name)

        ext = get_library_ext()
        net = _replace_suffix(ir_name, ext)
        # get input and output name
        ir_cfg = get_ir_config(deploy_cfg)
        backend_cfg = get_backend_config(deploy_cfg)
        input_names = ir_cfg['input_names']
        output_names = ir_cfg['output_names']
        weights = _replace_suffix(ir_name, '.txt')
        weights_path = osp.join(work_dir, weights)
        bytecode_path = _replace_suffix(ir_name, '.code')
        with open(weights_path, 'w') as f:
            f.write(','.join(input_names) + '\n')
            f.write(','.join(output_names) + '\n')
            use_vm = backend_cfg.model_inputs[0].get('use_vm', False)
            if use_vm:
                f.write(bytecode_path + '\n')

    precision = get_precision(deploy_cfg)
    dynamic_shape = is_dynamic_batch(deploy_cfg, input_name='input')
    return [
        dict(
            name=name,
            net=net,
            weights=weights,
            backend=backend,
            precision=precision,
            batch_size=1,
            dynamic_shape=dynamic_shape)
    ]


def get_inference_info(deploy_cfg: mmengine.Config, model_cfg: mmengine.Config,
                       work_dir: str, device: str) -> Dict:
    """Get the inference information for pipeline.json.

    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.
        model_cfg (mmengine.Config): The model config dict.
        work_dir (str): Work dir to save json files.
        device (str): The device passed in.

    Return:
        dict: Composed of the model name, type, module, input, output and
            input_map.
    """
    name, _ = get_model_name_customs(deploy_cfg, model_cfg, work_dir, device)
    ir_config = get_ir_config(deploy_cfg)
    backend = get_backend(deploy_cfg=deploy_cfg)
    if backend in (Backend.TORCHSCRIPT, Backend.RKNN):
        output_names = ir_config.get('output_names', None)
        if get_partition_config(deploy_cfg) is not None:
            output_names = get_partition_config(
                deploy_cfg)['partition_cfg'][0]['output_names']
        input_map = dict(img='#0')
        output_map = {name: f'#{i}' for i, name in enumerate(output_names)}
    else:
        input_names = ir_config.get('input_names', None)
        input_name = input_names[0] if input_names else 'input'
        input_map = dict(img=input_name)
        output_map = {}
    is_batched = is_dynamic_batch(deploy_cfg, input_name=input_map['img'])
    return_dict = dict(
        name=name,
        type='Task',
        module='Net',
        is_batched=is_batched,
        input=['prep_output'],
        output=['infer_output'],
        input_map=input_map,
        output_map=output_map)
    if 'use_vulkan' in deploy_cfg['backend_config']:
        return_dict['use_vulkan'] = deploy_cfg['backend_config']['use_vulkan']
    return return_dict


def get_preprocess(deploy_cfg: mmengine.Config, model_cfg: mmengine.Config,
                   device: str):
    task_processor = build_task_processor(
        model_cfg=model_cfg, deploy_cfg=deploy_cfg, device=device)
    transforms = task_processor.get_preprocess()

    if get_backend(deploy_cfg) == Backend.RKNN:
        del transforms[-2]
        for transform in transforms:
            if transform['type'] == 'Normalize':
                transform['to_float'] = False
                transform['mean'] = [0, 0, 0]
                transform['std'] = [1, 1, 1]
    if transforms[0]['type'] != 'Lift':
        assert transforms[0]['type'] == 'LoadImageFromFile', \
            'The first item type of pipeline should be LoadImageFromFile'
    return dict(
        type='Task',
        module='Transform',
        name='Preprocess',
        input=['img'],
        output=['prep_output'],
        transforms=transforms)


def get_postprocess(deploy_cfg: mmengine.Config, model_cfg: mmengine.Config,
                    work_dir: str, device: str, **kwargs) -> Dict:
    """Get the post process information for pipeline.json.

    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.
        model_cfg (mmengine.Config): The model config dict.
        work_dir (str): Work dir to save json files.
        device (str): The device passed in.
    Return:
        dict: Composed of the model name, type, module, input, params and
            output.
    """
    task_processor = build_task_processor(
        model_cfg=model_cfg, deploy_cfg=deploy_cfg, device=device)
    post_processor = task_processor.get_postprocess(work_dir)
    module = get_codebase(deploy_cfg).value
    module = 'mmdet' if module == 'mmyolo' else module
    module = 'mmcls' if module == 'mmpretrain' else module
    module = 'mmedit' if module == 'mmagic' else module
    # mmocr det models depend on postprocess from mmdet
    if module == 'mmocr' and post_processor['type'] == 'ResizeInstanceMask':
        module = 'mmdet'

    return dict(
        type='Task',
        module=module,
        name='postprocess',
        component=post_processor['type'],
        params=post_processor.get('params', dict()),
        output=['post_output'])


def get_deploy(deploy_cfg: mmengine.Config, model_cfg: mmengine.Config,
               work_dir: str, device: str) -> Dict:
    """Get the inference information for pipeline.json.

    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.
        model_cfg (mmengine.Config): The model config dict.
        work_dir (str): Work dir to save json files.
        device (str): The device passed in.

    Return:
        dict: Composed of version, task, models and customs.
    """

    task = get_task_type(deploy_cfg)
    cls_name = task_map[task]['cls_name']
    _, customs = get_model_name_customs(
        deploy_cfg, model_cfg, work_dir=work_dir, device=device)
    version = get_mmdeploy_version()
    models = get_models(deploy_cfg, model_cfg, work_dir, device)
    return dict(version=version, task=cls_name, models=models, customs=customs)


def get_pipeline(deploy_cfg: mmengine.Config, model_cfg: mmengine.Config,
                 work_dir: str, device: str) -> Dict:
    """Get the inference information for pipeline.json.

    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.
        model_cfg (mmengine.Config): The model config dict.
        work_dir (str): Work dir to save json files.
        device (str): The device passed in.
    Return:
        dict: Composed of input node name, output node name and the tasks.
    """
    preprocess = get_preprocess(deploy_cfg, model_cfg, device=device)
    infer_info = get_inference_info(
        deploy_cfg, model_cfg, work_dir=work_dir, device=device)
    postprocess = get_postprocess(
        deploy_cfg, model_cfg, work_dir, device=device)
    task = get_task_type(deploy_cfg)
    input_names = preprocess['input']
    output_names = postprocess['output']
    if task in [
            Task.CLASSIFICATION, Task.SUPER_RESOLUTION, Task.VIDEO_RECOGNITION
    ]:
        postprocess['input'] = infer_info['output']
    else:
        postprocess['input'] = preprocess['output'] + infer_info['output']

    return dict(
        pipeline=dict(
            input=input_names,
            output=output_names,
            tasks=[preprocess, infer_info, postprocess]))


def get_detail(deploy_cfg: mmengine.Config, model_cfg: mmengine.Config,
               pth: str) -> Dict:
    """Get the detail information for detail.json.

    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.
        model_cfg (mmengine.Config): The model config dict.
        pth (str): The checkpoint weight of pytorch model.
    Return:
        dict: Composed of version, codebase, codebase_config, onnx_config,
            backend_config and calib_config.
    """
    version = get_mmdeploy_version()
    codebase = get_task(deploy_cfg)
    codebase['pth'] = pth
    codebase['config'] = model_cfg.filename
    codebase_config = deploy_cfg.get('codebase_config', dict())
    ir_config = get_ir_config(deploy_cfg)
    backend_config = deploy_cfg.get('backend_config', dict())
    calib_config = deploy_cfg.get('calib_config', dict())
    return dict(
        version=version,
        codebase=codebase,
        codebase_config=codebase_config,
        onnx_config=ir_config,
        backend_config=backend_config,
        calib_config=calib_config)


def export2SDK(deploy_cfg: Union[str, mmengine.Config],
               model_cfg: Union[str, mmengine.Config], work_dir: str, pth: str,
               device: str, **kwargs):
    """Export information to SDK.

    This function dump `deploy.json`,
    `pipeline.json` and `detail.json` to work dir.
    Args:
        deploy_cfg (str | mmengine.Config): Deploy config file or dict.
        model_cfg (str | mmengine.Config): Model config file or dict.
        work_dir (str): Work dir to save json files.
        pth (str): The path of the model checkpoint weights.
        device (str): The device passed in.
    """
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
    deploy_info = get_deploy(deploy_cfg, model_cfg, work_dir, device)
    pipeline_info = get_pipeline(deploy_cfg, model_cfg, work_dir, device)
    detail_info = get_detail(deploy_cfg, model_cfg, pth=pth)

    mmengine.dump(
        deploy_info,
        '{}/deploy.json'.format(work_dir),
        sort_keys=False,
        indent=4)
    mmengine.dump(
        pipeline_info,
        '{}/pipeline.json'.format(work_dir),
        sort_keys=False,
        indent=4)
    mmengine.dump(
        detail_info,
        '{}/detail.json'.format(work_dir),
        sort_keys=False,
        indent=4)
