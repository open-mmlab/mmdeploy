# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import os
import os.path as osp

from packaging import version

REPO_NAMES = dict(
    mmcls='mmclassification',
    mmdet='mmdetection',
    mmseg='mmsegmentation',
    mmdet3d='mmdetection3d',
    mmedit='mmediting',
    mmocr='mmocr',
    mmpose='mmpose',
    mmrotate='mmrotate',
    mmaction='mmaction2',
    mmyolo='mmyolo')

MMDEPLOY_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))


def prepare_codebases(codebases, work_dir):
    work_dir = os.path.abspath(work_dir)
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
    for codebase in codebases:
        full_name = REPO_NAMES[codebase]
        target_dir = os.path.join(work_dir, full_name)
        ret = os.system(
            f'git clone --depth 1 '
            f'https://github.com/open-mmlab/{full_name}.git {target_dir}')
        assert ret == 0, f'Failed to clone codebase: {codebase}'
        if codebase == 'mmyolo':
            os.system(f'ln -sf {target_dir}/configs/deploy '
                      f'{MMDEPLOY_DIR}/configs/mmyolo')
    print(f'All codebases cloned to {work_dir}')


def install_torch(torch_version):
    cuda_version = os.environ.get('CUDA_VERSION', '11.3')
    cuda_int = ''.join(cuda_version.split('.')[:2])
    if version.parse(torch_version) < version.parse('1.10.0'):
        cuda_int = '111'
    is_torch_v2 = version.parse(torch_version) >= version.parse('2.0.0')
    tv_version = '0.15.1' if is_torch_v2 else f'0{torch_version[1:]}'
    if is_torch_v2:
        cmd = f'python -m pip install torch=={torch_version} ' \
              f'torchvision=={tv_version}'
    else:
        cmd = f'python -m pip install ' \
              f'torch=={torch_version}+cu{cuda_int} ' \
              f'torchvision=={tv_version}+cu{cuda_int} ' \
              f'-f https://download.pytorch.org/whl/torch_stable.html'
    ret = os.system(cmd)
    if ret != 0:
        print(f'Failed to install pytorch with command:\n{cmd}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch-version', type=str, help='Torch version')
    parser.add_argument('--codebases', type=str, help='Codebase names')
    parser.add_argument(
        '--work-dir', type=str, default='.', help='working directory')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.codebases = args.codebases.split(' ')
    assert len(args.codebases), 'at least input one codebases'
    install_torch(args.torch_version)
    prepare_codebases(args.codebases, args.work_dir)


if __name__ == '__main__':
    main()
