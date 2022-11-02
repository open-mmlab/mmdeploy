# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import re
from typing import Dict, List, Tuple, Union

import mmcv

from mmdeploy.apis import build_task_processor
from mmdeploy.utils import (Backend, Task, get_backend, get_codebase,
                            get_common_config, get_ir_config, get_root_logger,
                            get_task_type, is_dynamic_batch, load_config)
from mmdeploy.utils.constants import SDK_TASK_MAP as task_map
from .tracer import add_transform_tag, get_transform_static


def get_mmdpeloy_version() -> str:
    """Return the version of MMDeploy."""
    import mmdeploy
    version = mmdeploy.__version__
    return version


def get_task(deploy_cfg: mmcv.Config) -> Dict:
    """Get the task info for mmdeploy.json. The task info is composed of
    task_name, the codebase name and the codebase version.

    Args:
        deploy_cfg (mmcv.Config): Deploy config dict.

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


def get_model_name_customs(deploy_cfg: mmcv.Config, model_cfg: mmcv.Config,
                           work_dir: str, device: str) -> Tuple:
    """Get the model name and dump custom file.

    Args:
        deploy_cfg (mmcv.Config): Deploy config dict.
        model_cfg (mmcv.Config): The model config dict.
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
        from mmocr.models.builder import build_convertor
        label_convertor = model_cfg.model.label_convertor
        assert label_convertor is not None, 'model_cfg contains no label '
        'convertor'
        max_seq_len = 40  # default value in EncodeDecodeRecognizer of mmocr
        label_convertor.update(max_seq_len=max_seq_len)
        label_convertor = build_convertor(label_convertor)
        fd = open(f'{work_dir}/dict_file.txt', mode='w+')
        for item in label_convertor.idx2char:
            fd.write(item + '\n')
        fd.close()
        customs.append('dict_file.txt')
    return name, customs


def get_models(deploy_cfg: Union[str, mmcv.Config],
               model_cfg: Union[str, mmcv.Config], work_dir: str,
               device: str) -> List:
    """Get the output model informantion for deploy.json.

    Args:
        deploy_cfg (mmcv.Config): Deploy config dict.
        model_cfg (mmcv.Config): The model config dict.
        work_dir (str): Work dir to save json files.
        device (str): The device passed in.

    Return:
        list[dict]: The list contains dicts composed of the model name, net,
            weghts, backend, precision batchsize and dynamic_shape.
    """
    name, _ = get_model_name_customs(deploy_cfg, model_cfg, work_dir, device)
    precision = 'FP32'
    ir_name = get_ir_config(deploy_cfg)['save_file']
    net = ir_name
    weights = ''
    backend = get_backend(deploy_cfg=deploy_cfg)

    def replace_suffix(file_name: str, dst_suffix: str) -> str:
        """Replace the suffix to the destination one.

        Args:
            file_name (str): The file name to be operated.
            dst_suffix (str): The destination suffix.

        Return:
            str: The file name of which the suffix has been replaced.
        """
        return re.sub(r'\.[a-z]+', dst_suffix, file_name)

    if backend == Backend.TENSORRT:
        net = replace_suffix(ir_name, '.engine')
        common_cfg = get_common_config(deploy_cfg)
        fp16_mode = common_cfg.get('fp16_mode', False)
        int8_mode = common_cfg.get('int8_mode', False)
        if fp16_mode:
            precision = 'FP16'
        if int8_mode:
            precision = 'INT8'
    elif backend == Backend.PPLNN:
        precision = 'FP16'
        weights = replace_suffix(ir_name, '.json')
        net = ir_name
    elif backend == Backend.OPENVINO:
        net = replace_suffix(ir_name, '.xml')
        weights = replace_suffix(ir_name, '.bin')
    elif backend == Backend.NCNN:
        net = replace_suffix(ir_name, '.param')
        weights = replace_suffix(ir_name, '.bin')
        if 'precision' in deploy_cfg['backend_config']:
            precision = deploy_cfg['backend_config']['precision']
    elif backend == Backend.ASCEND:
        net = replace_suffix(ir_name, '.om')
    elif backend == Backend.SNPE:
        net = replace_suffix(ir_name, '.dlc')
    elif backend == Backend.RKNN:
        net = replace_suffix(ir_name, '.rknn')
    elif backend in [Backend.ONNXRUNTIME, Backend.TORCHSCRIPT]:
        pass
    elif backend == Backend.COREML:
        from mmdeploy.backend.coreml import get_model_suffix
        convert_to = deploy_cfg.backend_config.convert_to
        suffix = get_model_suffix(convert_to)
        net = replace_suffix(ir_name, suffix)
    else:
        raise NotImplementedError(f'Not supported backend: {backend.value}.')

    dynamic_shape = is_dynamic_batch(deploy_cfg, input_name='input')
    batch_size = 1
    return [
        dict(
            name=name,
            net=net,
            weights=weights,
            backend=backend.value,
            precision=precision,
            batch_size=batch_size,
            dynamic_shape=dynamic_shape)
    ]


def get_inference_info(deploy_cfg: mmcv.Config, model_cfg: mmcv.Config,
                       work_dir: str, device: str) -> Dict:
    """Get the inference information for pipeline.json.

    Args:
        deploy_cfg (mmcv.Config): Deploy config dict.
        model_cfg (mmcv.Config): The model config dict.
        work_dir (str): Work dir to save json files.
        device (str): The device passed in.

    Return:
        dict: Composed of the model name, type, module, input, output and
            input_map.
    """
    name, _ = get_model_name_customs(deploy_cfg, model_cfg, work_dir, device)
    type = 'Task'
    module = 'Net'
    input = ['prep_output']
    output = ['infer_output']
    ir_config = get_ir_config(deploy_cfg)

    backend = get_backend(deploy_cfg=deploy_cfg)
    if backend in (Backend.TORCHSCRIPT, Backend.RKNN):
        output_names = ir_config.get('output_names', None)
        input_map = dict(img='#0')
        output_map = {name: f'#{i}' for i, name in enumerate(output_names)}
    else:
        input_names = ir_config.get('input_names', None)
        input_name = input_names[0] if input_names else 'input'
        input_map = dict(img=input_name)
        output_map = {}
    return_dict = dict(
        name=name,
        type=type,
        module=module,
        input=input,
        output=output,
        input_map=input_map,
        output_map=output_map)
    if 'use_vulkan' in deploy_cfg['backend_config']:
        return_dict['use_vulkan'] = deploy_cfg['backend_config']['use_vulkan']
    return return_dict


def get_preprocess(deploy_cfg: mmcv.Config, model_cfg: mmcv.Config,
                   device: str):
    """Get the pre process information for pipeline.json.

    Args:
        deploy_cfg (mmcv.Config): Deploy config dict.
        model_cfg (mmcv.Config): The model config dict.
        device (str): The device passed in.
    """
    task_processor = build_task_processor(
        model_cfg=model_cfg, deploy_cfg=deploy_cfg, device=device)
    pipeline = task_processor.get_preprocess()
    type = 'Task'
    module = 'Transform'
    name = 'Preprocess'
    input = ['img']
    output = ['prep_output']
    meta_keys = [
        'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
        'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'valid_ratio'
    ]
    if 'transforms' in pipeline[-1]:
        transforms = pipeline[-1]['transforms']
        transforms.insert(0, pipeline[0])
        for transform in transforms:
            if transform['type'] == 'Resize':
                transform['size'] = pipeline[-1].img_scale[::-1]
                if 'img_scale' in transform:
                    transform.pop('img_scale')
    else:
        pipeline = [
            item for item in pipeline if item['type'] != 'MultiScaleFilpAug'
        ]
        transforms = pipeline
    transforms = [
        item for item in transforms if 'Random' not in item['type']
        and 'RescaleToZeroOne' not in item['type']
    ]
    for i, transform in enumerate(transforms):
        if 'keys' in transform and transform['keys'] == ['lq']:
            transform['keys'] = ['img']
        if 'key' in transform and transform['key'] == 'lq':
            transform['key'] = 'img'
        if transform['type'] == 'Collect':
            meta_keys += transform[
                'meta_keys'] if 'meta_keys' in transform else []
            transform['meta_keys'] = list(set(meta_keys))

    if get_backend(deploy_cfg) == Backend.RKNN:
        del transforms[-2]
        for transform in transforms:
            if transform['type'] == 'Normalize':
                transform['to_float'] = False
    assert transforms[0]['type'] == 'LoadImageFromFile', 'The first item type'\
        ' of pipeline should be LoadImageFromFile'

    return dict(
        type=type,
        module=module,
        name=name,
        input=input,
        output=output,
        transforms=transforms)


def get_postprocess(deploy_cfg: mmcv.Config, model_cfg: mmcv.Config,
                    device: str, **kwargs) -> Dict:
    """Get the post process information for pipeline.json.

    Args:
        deploy_cfg (mmcv.Config): Deploy config dict.
        model_cfg (mmcv.Config): The model config dict.
        device (str): The device passed in.

    Return:
        dict: Composed of the model name, type, module, input, params and
            output.
    """
    module = get_codebase(deploy_cfg).value
    type = 'Task'
    name = 'postprocess'
    task = get_task_type(deploy_cfg)
    task_processor = build_task_processor(
        model_cfg=model_cfg, deploy_cfg=deploy_cfg, device=device)
    params = task_processor.get_postprocess()

    # TODO remove after adding instance segmentation to task processor
    if task == Task.OBJECT_DETECTION and 'mask_thr_binary' in params:
        task = Task.INSTANCE_SEGMENTATION

    component = task_map[task]['component']
    if get_backend(deploy_cfg) == Backend.RKNN:
        if 'YOLO' in task_processor.model_cfg.model.type:
            bbox_head = task_processor.model_cfg.model.bbox_head
            component = bbox_head.type
            params['anchor_generator'] = bbox_head.get('anchor_generator',
                                                       None)
        else:  # default using base_dense_head
            component = 'BaseDenseHead'

    if task != Task.SUPER_RESOLUTION and task != Task.SEGMENTATION:
        if 'type' in params:
            component = params.pop('type')
    output = ['post_output']
    return dict(
        type=type,
        module=module,
        name=name,
        component=component,
        params=params,
        output=output)


def get_deploy(deploy_cfg: mmcv.Config, model_cfg: mmcv.Config, work_dir: str,
               device: str) -> Dict:
    """Get the inference information for pipeline.json.

    Args:
        deploy_cfg (mmcv.Config): Deploy config dict.
        model_cfg (mmcv.Config): The model config dict.
        work_dir (str): Work dir to save json files.
        device (str): The device passed in.

    Return:
        dict: Composed of version, task, models and customs.
    """

    task = get_task_type(deploy_cfg)
    cls_name = task_map[task]['cls_name']
    _, customs = get_model_name_customs(
        deploy_cfg, model_cfg, work_dir=work_dir, device=device)
    version = get_mmdpeloy_version()
    models = get_models(deploy_cfg, model_cfg, work_dir, device)
    return dict(version=version, task=cls_name, models=models, customs=customs)


def get_pipeline(deploy_cfg: mmcv.Config, model_cfg: mmcv.Config,
                 work_dir: str, device: str) -> Dict:
    """Get the inference information for pipeline.json.

    Args:
        deploy_cfg (mmcv.Config): Deploy config dict.
        model_cfg (mmcv.Config): The model config dict.
        work_dir (str): Work dir to save json files.
        device (str): The device passed in.

    Return:
        dict: Composed of input node name, output node name and the tasks.
    """
    preprocess = get_preprocess(deploy_cfg, model_cfg, device)
    infer_info = get_inference_info(deploy_cfg, model_cfg, work_dir, device)
    postprocess = get_postprocess(deploy_cfg, model_cfg, device)
    task = get_task_type(deploy_cfg)
    input_names = preprocess['input']
    output_names = postprocess['output']
    if task == Task.CLASSIFICATION or task == Task.SUPER_RESOLUTION:
        postprocess['input'] = infer_info['output']
    else:
        postprocess['input'] = preprocess['output'] + infer_info['output']

    return dict(
        pipeline=dict(
            input=input_names,
            output=output_names,
            tasks=[preprocess, infer_info, postprocess]))


def get_detail(deploy_cfg: mmcv.Config, model_cfg: mmcv.Config,
               pth: str) -> Dict:
    """Get the detail information for detail.json.

    Args:
        deploy_cfg (mmcv.Config): Deploy config dict.
        model_cfg (mmcv.Config): The model config dict.
        pth (str): The checkpoint weight of pytorch model.

    Return:
        dict: Composed of version, codebase, codebase_config, onnx_config,
            backend_config and calib_config.
    """
    version = get_mmdpeloy_version()
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


def export2SDK(deploy_cfg: Union[str, mmcv.Config],
               model_cfg: Union[str, mmcv.Config], work_dir: str, pth: str,
               device: str, **kwargs):
    """Export information to SDK. This function dump `deploy.json`,
    `pipeline.json` and `detail.json` to work dir.

    Args:
        deploy_cfg (str | mmcv.Config): Deploy config file or dict.
        model_cfg (str | mmcv.Config): Model config file or dict.
        work_dir (str): Work dir to save json files.
        pth (str): The path of the model checkpoint weights.
    """
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
    deploy_info = get_deploy(deploy_cfg, model_cfg, work_dir, device)
    pipeline_info = get_pipeline(deploy_cfg, model_cfg, work_dir, device)
    detail_info = get_detail(deploy_cfg, model_cfg, pth=pth)
    transform_static, tag = get_transform_static(
        pipeline_info['pipeline']['tasks'][0]['transforms'])
    pipeline_info = add_transform_tag(pipeline_info, tag)
    mmcv.dump(
        deploy_info,
        '{}/deploy.json'.format(work_dir),
        sort_keys=False,
        indent=4)
    mmcv.dump(
        pipeline_info,
        '{}/pipeline.json'.format(work_dir),
        sort_keys=False,
        indent=4)
    mmcv.dump(
        detail_info,
        '{}/detail.json'.format(work_dir),
        sort_keys=False,
        indent=4)
