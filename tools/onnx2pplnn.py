# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import collections
import logging

from mmdeploy.apis.pplnn import onnx2pplnn


def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX to PPLNN.')
    parser.add_argument('onnx_path', help='ONNX model path')
    parser.add_argument(
        'output_path', help='output PPLNN algorithm path in json format')
    parser.add_argument(
        '--device',
        help='`the device of model during conversion',
        default='cuda:0')
    parser.add_argument(
        '--opt-shapes',
        help='`Optical shapes for PPLNN optimization. The shapes must be able'
        'to be evaluated by python, e,g., `[1, 3, 224, 224]`',
        default='[1, 3, 224, 224]')
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logging.basicConfig(
        format='%(asctime)s,%(name)s %(levelname)-8s'
        ' [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(args.log_level)

    onnx_path = args.onnx_path
    output_path = args.output_path
    device = args.device

    input_shapes = eval(args.opt_shapes)
    assert isinstance(
        input_shapes, collections.Sequence), \
        'The opt-shape must be a sequence.'
    assert isinstance(input_shapes[0], int) or (isinstance(
        input_shapes[0], collections.Sequence)), \
        'The opt-shape must be a sequence of int or a sequence of sequence.'
    if isinstance(input_shapes[0], int):
        input_shapes = [input_shapes]

    logging.info(f'onnx2ppl: \n\tonnx_path: {onnx_path} '
                 f'\n\toutput_path: {output_path}'
                 f'\n\topt_shapes: {input_shapes}')
    try:
        onnx2pplnn(output_path, onnx_path, device, input_shapes)
        logging.info('onnx2tpplnn success.')
    except Exception as e:
        logging.error(e)
        logging.error('onnx2tpplnn failed.')


if __name__ == '__main__':
    main()
