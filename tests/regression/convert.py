# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from enum import Enum

import yaml
from easydict import EasyDict as edict

from mmdeploy.utils import get_backend, get_task_type, load_config


class Tree(Enum):
    mmcls = 'tree/1.x',
    mmdet = 'tree/3.x',
    mmdet3d = 'tree/1.x',
    mmedit = 'tree/1.x',
    mmocr = 'tree/1.x',
    mmpose = 'tree/1.x',
    mmrotate = 'tree/1.x',
    mmseg = 'tree/1.x'


def parse_args():
    parser = argparse.ArgumentParser(description='From yaml export markdown')
    parser.add_argument('yml_file', help='yml config path')
    parser.add_argument('output', help='Output markdown file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert osp.exists(args.yml_file), f'File not exists: {args.yml_file}'
    output_dir, _ = osp.split(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    backends = [
        'onnxruntime', 'tensorrt', 'torchscript', 'pplnn', 'openvino', 'ncnn'
    ]
    header = ['model', 'task'] + backends
    aligner = [':--'] * 2 + [':--'] * len(backends)

    def write_row_f(writer, row):
        writer.write('|' + '|'.join(row) + '|\n')

    print(f'Processing{args.yml_file}')
    with open(args.yml_file, 'r') as reader, open(args.output, 'w') as writer:
        config = yaml.load(reader, Loader=yaml.FullLoader)
        config = edict(config)
        write_row_f(writer, header)
        write_row_f(writer, aligner)
        repo_url = config.globals.repo_url
        (head, tail) = osp.split(args.yml_file)
        (head, tail) = osp.splitext(tail)
        tree = Tree[head].value[0]
        print(tree)
        for i in range(len(config.models)):
            name = config.models[i].name
            model_configs = config.models[i].model_configs
            pipelines = config.models[i].pipelines
            config_url = osp.join(repo_url, tree, model_configs[0])
            config_url, _ = osp.split(config_url)
            support_backends = {b: 'N' for b in backends}
            deploy_config = [
                pipelines[i].deploy_config for i in range(len(pipelines))
            ]
            cfg = [
                load_config(deploy_config[i])
                for i in range(len(deploy_config))
            ]
            task = [
                get_task_type(cfg[i][0]).value
                for i in range(len(deploy_config))
            ]
            backend_type = [
                get_backend(cfg[i][0]).value
                for i in range(len(deploy_config))
            ]
            for i in range(len(deploy_config)):
                support_backends[backend_type[i]] = 'Y'
            support_backends = [support_backends[i] for i in backends]
            model_name = f'[{name}]({config_url})'
            row = [model_name, task[i]] + support_backends

            write_row_f(writer, row)
            print(f'Save to {args.output}')


if __name__ == '__main__':
    main()
