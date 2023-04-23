# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import yaml
from mmengine import Config

from mmdeploy.utils import get_backend, get_task_type, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='from yaml export markdown table')
    parser.add_argument('yml_file', help='input yml config path')
    parser.add_argument('output', help='output markdown file path')
    parser.add_argument(
        '--backends',
        nargs='+',
        help='backends you want to generate',
        default=[
            'onnxruntime', 'tensorrt', 'torchscript', 'pplnn', 'openvino',
            'ncnn'
        ])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert osp.exists(args.yml_file), f'File not exists: {args.yml_file}'
    output_dir, _ = osp.split(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    header = ['model', 'task'] + args.backends
    aligner = [':--'] * 2 + [':--:'] * len(args.backends)

    def write_row_f(writer, row):
        writer.write('|' + '|'.join(row) + '|\n')

    print(f'Processing{args.yml_file}')
    with open(args.yml_file, 'r') as reader, open(args.output, 'w') as writer:
        config = yaml.load(reader, Loader=yaml.FullLoader)
        config = Config(config)
        write_row_f(writer, header)
        write_row_f(writer, aligner)
        repo_url = config.globals.repo_url
        for i in range(len(config.models)):
            name = config.models[i].name
            model_configs = config.models[i].model_configs
            pipelines = config.models[i].pipelines
            config_url = osp.join(repo_url, model_configs[0])
            config_url, _ = osp.split(config_url)
            support_backends = {b: 'N' for b in args.backends}
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
            support_backends = [support_backends[i] for i in args.backends]
            model_name = f'[{name}]({config_url})'
            row = [model_name, task[i]] + support_backends

            write_row_f(writer, row)
        print(f'Save to {args.output}')


if __name__ == '__main__':
    main()
