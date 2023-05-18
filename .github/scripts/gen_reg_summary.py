# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import pandas as pd

MMDEPLOY_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
MMDEPLOY_URL = 'https://github.com/open-mmlab/mmdeploy/tree/main'

REPO_NAMES = dict(
    mmpretrain='mmpretrain',
    mmdet='mmdetection',
    mmseg='mmsegmentation',
    mmdet3d='mmdetection3d',
    mmagic='mmagic',
    mmocr='mmocr',
    mmpose='mmpose',
    mmrotate='mmrotate',
    mmaction='mmaction2',
    mmyolo='mmyolo')


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
    codebase_fullname = REPO_NAMES[codebase]
    columns = report.columns.tolist()
    convert_col_name = 'Conversion Result'
    convert_col_index = columns.index(convert_col_name)
    metric_col_names = columns[convert_col_index + 1:-1]
    repo_url = f'https://github.com/open-mmlab/{codebase_fullname}/tree/main'
    header = ['Index', 'Task', 'Model', 'Backend', 'Precision']
    header += metric_col_names
    header += ['Conversion', 'Benchmark', 'Pass:link:']
    aligner = [':-:'] * len(header)
    pass_flag = ':heavy_check_mark:'
    fail_flag = ':x:'
    with open(markdown_path, 'w') as f:
        write_row_f(f, header)
        write_row_f(f, aligner)
        counter = 0
        for idx, row in report.iterrows():
            backend = row['Backend']
            precision_type = row['Precision Type']
            model_cfg = row['Model Config'].split('configs')[1].replace(
                osp.sep, r'/')
            config_url = f'{repo_url}/configs/{model_cfg}'
            model = f'[{row["Model"]}]({config_url})'
            metric_col_values = [row[_] for _ in metric_col_names]
            metric_col_values = [
                _ if isinstance(_, str) else f'{_:.4f}'
                for _ in metric_col_values
            ]
            run_test = not all([str(_) == '-' for _ in metric_col_values])
            if backend.lower() == 'pytorch':
                counter += 1
                backend = f'[{backend}]({config_url})'
                conversion = pass_flag
                test = pass_flag
                benchmark_pass = pass_flag
                precision_type = 'fp32'
            else:
                ckpt = row['Checkpoint']
                ckpt_prefix = 'mmdeploy_regression_dir'
                assert ckpt.startswith(ckpt_prefix)
                deploy_cfg = row['Deploy Config'].replace(osp.sep, r'/')
                deploy_url = f'{MMDEPLOY_URL}/{deploy_cfg}'
                backend = f'[{backend}]({deploy_url})'
                parent, filename = osp.split(ckpt)
                if '.' in filename:
                    ckpt = parent
                ckpt = ckpt.replace(osp.sep, r'/')
                current_url = ckpt.replace(ckpt_prefix, args.url_prefix)
                convert_pass = eval(row[convert_col_name])
                test_pass = eval(row['Test Pass'])
                if convert_pass:
                    conversion = pass_flag
                    if test_pass:
                        if run_test:
                            benchmark_pass = pass_flag
                        else:
                            benchmark_pass = ':white_check_mark:'
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
            line = [f'{counter}', row['Task'], model, backend, precision_type]
            line += metric_col_values
            line += [conversion, benchmark_pass, test]
            write_row_f(f, line)

    print(f'Saved to {markdown_path}')


if __name__ == '__main__':
    main()
