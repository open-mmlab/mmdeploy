# Copyright (c) MegFlow. All rights reserved.
# /bin/python3

import argparse
import os
import re


def make_parser():
    parser = argparse.ArgumentParser('Doc link checker')
    parser.add_argument(
        '--http', default=False, type=bool, help='check http or not ')
    parser.add_argument(
        '--dir', default='./docs', type=str, help='the directory to be check')
    return parser


pattern = re.compile(r'\[.*?\]\(.*?\)')


def analyze_doc(home, path):
    problem_list = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            if '[' in line and ']' in line and '(' in line and ')' in line:
                all = pattern.findall(line)
                for item in all:
                    start = item.find('(')
                    end = item.find(')')
                    ref = item[start + 1:end]

                    if ref.startswith('http') or ref.startswith('#'):
                        continue
                    if '.md#' in ref:
                        ref = ref[ref.find('#'):]
                    fullpath = os.path.join(home, ref)
                    if not os.path.exists(fullpath):
                        problem_list.append(ref)
            else:
                continue
    if len(problem_list) > 0:
        print(f'{path}:')
        for item in problem_list:
            print(f'\t {item}')
        print('\n')


def traverse(_dir):
    for home, dirs, files in os.walk(_dir):
        if './target' in home or './.github' in home:
            continue
        for filename in files:
            if filename.endswith('.md'):
                path = os.path.join(home, filename)
                if os.path.islink(path) is False:
                    analyze_doc(home, path)


if __name__ == '__main__':
    args = make_parser().parse_args()
    traverse(args.dir)
