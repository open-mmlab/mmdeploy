# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os.path as osp
from copy import deepcopy
from typing import Optional, Sequence

import h5py
import tqdm
from mmengine import Config

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_root_logger, load_config


def get_tensor_func(model, input_data):
    input_data = model.data_preprocessor(input_data)
    return input_data['inputs']


def process_model_config(model_cfg: Config,
                         input_shape: Optional[Sequence[int]] = None):
    """Process the model config.

    Args:
        model_cfg (Config): The model config.
        input_shape (list[int]): A list of two integer in (width, height)
            format specifying input shape. Default: None.

    Returns:
        Config: the model config after processing.
    """

    cfg = model_cfg.copy()

    pipeline = cfg.test_pipeline

    for i, transform in enumerate(pipeline):
        # for static exporting
        if transform.type == 'Resize':
            pipeline[i].keep_ratio = False
            pipeline[i].scale = tuple(input_shape)
        if transform.type in ('YOLOv5KeepRatioResize', 'LetterResize'):
            pipeline[i].scale = tuple(input_shape)

    pipeline = [
        transform for transform in pipeline
        if transform.type != 'LoadAnnotations'
    ]
    cfg.test_pipeline = pipeline
    return cfg


def get_quant(deploy_cfg: Config,
              model_cfg: Config,
              shape_dict: dict,
              checkpoint_path: str,
              work_dir: str,
              device: str = 'cpu',
              dataset_type: str = 'val'):

    model_shape = list(shape_dict.values())[0]
    model_cfg = process_model_config(model_cfg,
                                     (model_shape[3], model_shape[2]))

    task_processor = build_task_processor(model_cfg, deploy_cfg, device)
    model = task_processor.build_pytorch_model(checkpoint_path)
    calib_dataloader = deepcopy(model_cfg[f'{dataset_type}_dataloader'])
    calib_dataloader['batch_size'] = 1

    dataloader = task_processor.build_dataloader(calib_dataloader)
    output_quant_dataset_path = osp.join(work_dir, 'calib_data.h5')

    with h5py.File(output_quant_dataset_path, mode='w') as file:
        calib_data_group = file.create_group('calib_data')
        input_data_group = calib_data_group.create_group('input')

        # get an available input shape randomly
        for data_id, input_data in enumerate(tqdm.tqdm(dataloader)):
            # input_data = data_preprocessor(input_data)['inputs'].numpy()
            input_data = get_tensor_func(model, input_data).numpy()
            calib_data_shape = input_data.shape
            assert model_shape[2] >= calib_data_shape[2] and model_shape[
                3] >= calib_data_shape[
                    3], f'vacc backend model shape is {tuple(model_shape[2:])}, \
                        the calib_data shape {calib_data_shape[2:]} is bigger'

            input_data_group.create_dataset(
                str(data_id),
                shape=input_data.shape,
                compression='gzip',
                compression_opts=4,
                data=input_data)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate vacc quant dataset from ONNX.')
    parser.add_argument('--deploy-cfg', help='Input deploy config path')
    parser.add_argument('--model-cfg', help='Input model config path')
    parser.add_argument('--shape-dict', help='Input model shape')
    parser.add_argument('--checkpoint-path', help='checkpoint path')
    parser.add_argument('--work-dir', help='Output quant dataset dir')

    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger = get_root_logger(log_level=args.log_level)

    deploy_cfg, model_cfg = load_config(args.deploy_cfg, args.model_cfg)
    work_dir = args.work_dir
    checkpoint_path = args.checkpoint_path
    shape_dict = eval(args.shape_dict)

    get_quant(deploy_cfg, model_cfg, shape_dict, checkpoint_path, work_dir)
    logger.info('onnx2vacc_quant_dataset success.')


if __name__ == '__main__':
    main()
