# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os.path as osp

import onnx
import onnx.helper

from mmdeploy.apis.onnx import extract_partition
from mmdeploy.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract model based on markers.')
    parser.add_argument('input_model', help='Input ONNX model')
    parser.add_argument('output_model', help='Output ONNX model')
    parser.add_argument(
        '--start',
        help='Start markers, format: func:type, e.g. backbone:input')
    parser.add_argument('--end', help='End markers')
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    args = parser.parse_args()

    args.start = args.start.split(',') if args.start else []
    args.end = args.end.split(',') if args.end else []

    return args


def collect_avaiable_marks(model):
    marks = []
    for node in model.graph.node:
        if node.op_type == 'Mark':
            for attr in node.attribute:
                if attr.name == 'func':
                    func = str(onnx.helper.get_attribute_value(attr), 'utf-8')
            if func not in marks:
                marks.append(func)
    return marks


def main():
    args = parse_args()

    logger = get_root_logger(log_level=args.log_level)

    model = onnx.load(args.input_model)
    marks = collect_avaiable_marks(model)
    logger.info('Available marks:\n    {}'.format('\n    '.join(marks)))

    extracted_model = extract_partition(model, args.start, args.end)

    if osp.splitext(args.output_model)[-1] != '.onnx':
        args.output_model += '.onnx'
    onnx.save(extracted_model, args.output_model)


if __name__ == '__main__':
    main()
