# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import pandas as pd
import yaml

MMDEPLOY_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
MMDEPLOY_URL = 'https://github.com/open-mmlab/mmdeploy/tree/main'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check Regression Test Result')
    parser.add_argument('report_path', help='report excel path')
    parser.add_argument('url_prefix', help='result url prefix')
    args = parser.parse_args()
    return args


def write_row_f(writer, row):
    writer.write('|' + '|'.join(row) + '|\n')


def main():
    args = parse_args()
    if not osp.exists(args.report_path):
        print(f'File not exists: {args.report_path}')
        return
    report = pd.read_excel(args.report_path)
    markdown_path = osp.splitext(args.report_path)[0] + '.md'
    _, fname = osp.split(args.report_path)
    codebase, _ = fname.split('_')
    test_yml = osp.join(MMDEPLOY_DIR, f'tests/regression/{codebase}.yml')
    with open(test_yml, 'r') as f:
        yml_cfg = yaml.load(f, Loader=yaml.FullLoader)
        repo_url = yml_cfg['globals']['repo_url']
    header = [
        'Index', 'Task', 'Model', 'Backend', 'Precision', 'Conversion',
        'Benchmark', 'Pass'
    ]
    aligner = [':-:'] * len(header)
    pass_flag = ':heavy_check_mark:'
    fail_flag = ':x:'
    with open(markdown_path, 'w') as f:
        write_row_f(f, header)
        write_row_f(f, aligner)
        counter = 0
        for idx, row in report.iterrows():
            backend = row['Backend']
            if backend.lower() == 'pytorch':
                continue
            ckpt = row['Checkpoint']
            ckpt_prefix = 'mmdeploy_regression_dir'
            assert ckpt.startswith(ckpt_prefix)
            model_cfg = row['Model Config'].split('configs')[1].replace(
                osp.sep, r'/')
            config_url = f'{repo_url}/configs/{model_cfg}'
            deploy_cfg = row['Deploy Config'].replace(osp.sep, r'/')
            deploy_url = f'{MMDEPLOY_URL}/{deploy_cfg}'
            backend = f'[{backend}]({deploy_url})'
            model = f'[{row["Model"]}]({config_url})'
            parent, filename = osp.split(ckpt)
            if '.' in filename:
                ckpt = parent
            ckpt = ckpt.replace(osp.sep, r'/')
            current_url = ckpt.replace(ckpt_prefix, args.url_prefix)
            convert_pass = eval(row['Conversion Result'])
            test_pass = eval(row['Test Pass'])
            if convert_pass:
                conversion = pass_flag
                if test_pass:
                    benchmark_pass = ':heavy_check_mark:'
                else:
                    benchmark_pass = fail_flag
            else:
                conversion = fail_flag
                benchmark_pass = ':o:'
            if test_pass:
                test = pass_flag
            else:
                test = fail_flag
            test = f'[{test}]({current_url})'
            line = [
                f'{counter}', row['Task'], model, backend,
                row['Precision Type'], conversion, benchmark_pass, test
            ]
            write_row_f(f, line)
            counter += 1

    print(f'Saved to {markdown_path}')


if __name__ == '__main__':
    main()
