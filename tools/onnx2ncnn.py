# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging

from mmdeploy.apis.ncnn import onnx2ncnn
from mmdeploy.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX to ncnn.')
    parser.add_argument('onnx_path', help='ONNX model path')
    parser.add_argument('output_param', help='output ncnn param path')
    parser.add_argument('output_bin', help='output bin path')
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
    output_param = args.output_param
    output_bin = args.output_bin

    logger.info(f'onnx2ncnn: \n\tonnx_path: {onnx_path} ')
    try:
        onnx2ncnn(onnx_path, output_param, output_bin)
        logger.info('onnx2ncnn success.')
    except Exception as e:
        logger.error(e)
        logger.error('onnx2ncnn failed.')


if __name__ == '__main__':
    main()
