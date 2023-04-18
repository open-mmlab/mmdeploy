# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import logging
import os
import os.path as osp
import shutil
import subprocess

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


def run_cmd(cmd_lines, log_path=None, raise_error=True):
    """
    Args:
        cmd_lines: (list[str]): A command in multiple line style.
        log_path (str): Path to log file.
        raise_error (bool): Whether to raise error when running cmd fails.
    """
    import platform
    system = platform.system().lower()

    if system == 'windows':
        sep = r'`'
    else:  # 'Linux', 'Darwin'
        sep = '\\'
    cmd_for_run = ' '.join(cmd_lines)
    cmd_for_log = f' {sep}\n'.join(cmd_lines) + '\n'
    if log_path is None:
        log_path = osp.join(MMDEPLOY_DIR, 'prepare_reg_test.log')
    log_dir, _ = osp.split(log_path)
    os.makedirs(log_dir, exist_ok=True)
    logging.info(100 * '-')
    logging.info(f'Start running cmd\n{cmd_for_log}')
    logging.info(f'Logging log to \n{log_path}')

    with open(log_path, 'a', encoding='utf-8') as file_handler:
        # write cmd
        file_handler.write(f'Command:\n{cmd_for_log}\n')
        file_handler.flush()
        process_res = subprocess.Popen(
            cmd_for_run,
            cwd=MMDEPLOY_DIR,
            shell=True,
            stdout=file_handler,
            stderr=file_handler)
        process_res.wait()
        return_code = process_res.returncode

    if return_code != 0:
        logging.error(f'Got shell return code={return_code}')
        with open(log_path, 'r') as f:
            content = f.read()
            logging.error(f'Log error message\n{content}')
        if raise_error:
            raise RuntimeError(f'Failed to run cmd:\n{cmd_for_run}')


def prepare_codebases(codebases):
    for codebase in codebases:
        full_name = REPO_NAMES[codebase]
        target_dir = os.path.join(MMDEPLOY_DIR, '..', full_name)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        cmd = [
            'git clone --depth 1 ',
            f'https://github.com/open-mmlab/{full_name}.git '
            f'{target_dir} '
        ]
        run_cmd(cmd)
        run_cmd([f'python -m mim install -r {target_dir}/requirements.txt'])
        run_cmd([f'python -m mim install -e {target_dir}'])
        if codebase == 'mmyolo':
            cmd = [
                f'cp -r {target_dir}/configs/deploy ',
                f'{MMDEPLOY_DIR}/configs/mmyolo '
            ]
            run_cmd(cmd)
            cmd = [
                f'cp {target_dir}/tests/regression/mmyolo.yml ',
                f'{MMDEPLOY_DIR}/tests/regression/mmyolo.yml '
            ]
            run_cmd(cmd)


def install_torch(torch_version):
    cuda_version = os.environ.get('CUDA_VERSION', '11.3')
    cuda_int = ''.join(cuda_version.split('.')[:2])
    if version.parse(torch_version) < version.parse('1.10.0'):
        cuda_int = '111'
    is_torch_v2 = version.parse(torch_version) >= version.parse('2.0.0')
    if is_torch_v2:
        tv_version = '0.15.1'
    else:
        ver = version.parse(torch_version)
        tv_version = f'0.{ver.minor+1}.{ver.micro}'
    if is_torch_v2:
        cmd = [
            f'python -m pip install torch=={torch_version} ',
            f'torchvision=={tv_version} '
        ]
    else:
        url = 'https://download.pytorch.org/whl/torch_stable.html'
        cmd = [
            'python -m pip install ', f'torch=={torch_version}+cu{cuda_int} ',
            f'torchvision=={tv_version}+cu{cuda_int} ', f'-f {url}'
        ]
    run_cmd(cmd)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch-version', type=str, help='Torch version')
    parser.add_argument(
        '--codebases', type=str, nargs='+', help='Codebase names')
    parser.add_argument(
        '--work-dir', type=str, default='.', help='working directory')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert len(args.codebases) > 0, 'at least input one codebases'
    install_torch(args.torch_version)
    prepare_codebases(args.codebases)


if __name__ == '__main__':
    main()
