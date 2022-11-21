# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import logging
import os
import os.path as osp

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Regression Test')
    parser.add_argument('regression_dir')
    parser.add_argument('--url-prefix', default='http://10.1.52.36:8989')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    regression_dir = args.regression_dir
    assert osp.exists(regression_dir)
    codebase_dirs = glob.glob(osp.join(regression_dir, 'mm*'))
    codebase_dirs = [d for d in codebase_dirs if os.path.isdir(d)]
    test_stats_path = osp.join(regression_dir, 'stats_report.xlsx')
    test_stats_writer = pd.ExcelWriter(test_stats_path)
    for i, cb_dir in enumerate(codebase_dirs):
        _, codebase_name = osp.split(cb_dir)
        # cb_log = cb_dir + '.log'
        torch_versions = [
            d for d in os.listdir(cb_dir) if d.startswith('torch')
        ]
        report_detail_path = osp.join(cb_dir, 'report_detail.xlsx')
        stats = []
        with pd.ExcelWriter(report_detail_path) as writer:
            for tv in torch_versions:
                version = tv.replace('torch', '')
                tv_dir = osp.join(cb_dir, tv)
                report_excel_path = osp.join(tv_dir,
                                             f'{codebase_name}_report.xlsx')
                if not osp.exists(report_excel_path):
                    logging.error(
                        f'Report file not found: {report_excel_path}')
                    continue
                report = pd.read_excel(report_excel_path)
                test_pass_key = report.columns[-1]
                report_failed = report.loc[report[test_pass_key].isin(
                    ['False', '-'])]
                tmp_report = report_failed.loc[report_failed[test_pass_key] ==
                                               'False']  # noqa
                num_failed = len(tmp_report)
                if num_failed == 0:
                    continue
                model_failed = set(tmp_report['Model'])
                report_failed = report_failed.loc[report_failed['Model'].isin(
                    model_failed)]
                model_failed_with_backend = set()
                for idx, row in tmp_report.iterrows():
                    s = f'{row["Model"]}+{row["Backend"]}'
                    model_failed_with_backend.add(s)
                model_failed_str = ' || '.join(list(model_failed_with_backend))
                stats.append([version, num_failed, model_failed_str])
                url_prefix = tv_dir.replace('/data2/regression_log',
                                            args.url_prefix)

                def add_url(row):
                    url = '-'
                    if str(row[test_pass_key]) == 'False':
                        ckpt = row['Checkpoint']
                        if ckpt != 'x':
                            url = osp.split(ckpt)[0].replace(
                                '${WORK_DIR}', url_prefix)
                        else:
                            url = url_prefix
                    return url

                report_failed['LOG_URL'] = report_failed.apply(add_url, axis=1)
                report_failed.to_excel(writer, sheet_name=version, index=False)

        df_stats = pd.DataFrame(
            stats, columns=['Torch', 'Failed Number', 'Failed Model Name'])
        df_stats.to_excel(
            test_stats_writer, sheet_name=codebase_name, index=False)
        print(f'Save results to {report_detail_path}')

    test_stats_writer.close()


if __name__ == '__main__':
    main()
