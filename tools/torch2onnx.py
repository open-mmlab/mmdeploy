# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os.path as osp

from mmdeploy.apis import torch2onnx
from mmdeploy.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX.')
    parser.add_argument('deploy_cfg', help='deploy config path')
    parser.add_argument('model_cfg', help='model config path')
    parser.add_argument('checkpoint', help='model checkpoint path')
    parser.add_argument('img', help='image used to convert model model')
    parser.add_argument('output', help='output onnx path')
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
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

    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg
    checkpoint_path = args.checkpoint
    img = args.img
    output_path = args.output
    work_dir, save_file = osp.split(output_path)
    device = args.device

    logger.info(f'torch2onnx: \n\tmodel_cfg: {model_cfg_path} '
                f'\n\tdeploy_cfg: {deploy_cfg_path}')
    try:
        torch2onnx(
            img,
            work_dir,
            save_file,
            deploy_cfg=deploy_cfg_path,
            model_cfg=model_cfg_path,
            model_checkpoint=checkpoint_path,
            device=device)
        logger.info('torch2onnx success.')
    except Exception as e:
        logger.error(e)
        logger.error('torch2onnx failed.')


if __name__ == '__main__':
    main()
