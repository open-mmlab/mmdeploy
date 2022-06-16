# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import glob
import logging
import os.path as osp


def check_index_rst(index_file: str):
    """
    Check if all md file in index.rst file exist.
    Give warning message if md file not added in index.rst
    Args:
        index_file(str): Path of index.rst file.

    Returns:
        bool: True if all md file exist and False if any md file not exists.
    """
    assert osp.exists(index_file), f'File not exists: {index_file}'
    work_dir = osp.dirname(index_file)
    md_files = glob.glob(osp.join(work_dir, '**/*.md'), recursive=True)
    md_files_rst = []
    all_file_exist = True
    with open(index_file, 'r') as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if line.endswith('.md'):
                path = osp.join(work_dir, line)
                md_files_rst.append(path)
                if not osp.exists(path):
                    all_file_exist = False
                    logging.error(f'File {line} not exists in '
                                  f'Line {idx} of index_file {index_file}')

    # check if all md files added in index.rst index_file
    rest_md_files = list(set(md_files) - set(md_files_rst))
    if len(rest_md_files) > 0:
        for f in rest_md_files:
            logging.warning(f'Please check whether file {f} '
                            f'should be added in {index_file}')
    return all_file_exist


def parse_args():
    parser = argparse.ArgumentParser(description='Check index.rst.')
    parser.add_argument(
        'file', type=str, help='Path of index.rst to be checked')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    success_flag = check_index_rst(args.file)
    if not success_flag:
        raise RuntimeError(
            f'Check fails with {args.file}. Please check logging message.')


if __name__ == '__main__':
    main()
