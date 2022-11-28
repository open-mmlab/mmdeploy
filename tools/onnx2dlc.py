# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging

from mmdeploy.apis.snpe import from_onnx
from mmdeploy.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ONNX to snpe dlc format.')
    parser.add_argument('onnx_path', help='ONNX model path')
    parser.add_argument('output_prefix', help='output snpe dlc model path')
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

    onnx_path = args.onnx_path
    output_prefix = args.output_prefix

    logger.info(f'onnx2dlc: \n\tonnx_path: {onnx_path} ')
    from_onnx(onnx_path, output_prefix)
    logger.info('onnx2dlc success.')


if __name__ == '__main__':
    main()
