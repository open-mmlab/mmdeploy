# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import logging
import os
import os.path as osp
import shutil
import tarfile
from glob import glob
from subprocess import CalledProcessError, run
from typing import Dict

import yaml

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGING_DIR = osp.join(CUR_DIR, 'packaging')


def _remove_if_exist(path):
    if osp.exists(path):
        logging.info(f'Remove path: {path}.')
        if osp.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def _copy(src_path, dst_path):
    assert osp.exists(src_path), f'src path: {src_path} not exist.'

    logging.info(f'copy path: {src_path} to {dst_path}.')
    if osp.isdir(src_path):
        if osp.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path)
    else:
        shutil.copy(src_path, dst_path)


def _call_command(cmd, cwd, stdout=None, stderr=None):
    if cmd == '':
        return
    logging.info(f'Process cmd: {cmd}')
    logging.info(f'work_path: {cwd}')
    run(cmd, stdout=stdout, stderr=stderr, cwd=cwd, shell=True)


def _create_tar(path, tar_name):
    logging.info(f'create tar file: {tar_name}')
    with tarfile.open(tar_name, 'w:gz') as tar:
        tar.add(path, arcname=os.path.basename(path))


def clear_mmdeploy(mmdeploy_dir: str):
    logging.info(f'cleaning mmdeploy: {mmdeploy_dir}')

    def _remove_in_mmdeploy(path):
        remove_dir = osp.join(mmdeploy_dir, path)
        _remove_if_exist(remove_dir)

    # remove build file
    _remove_in_mmdeploy('build')

    # remove dist
    _remove_in_mmdeploy('dist')

    # remove installed library
    _remove_in_mmdeploy('mmdeploy/lib')

    # remove onnx2ncnn and ncnn ext
    _remove_in_mmdeploy('mmdeploy/backend/ncnn/onnx2ncnn')
    _remove_in_mmdeploy('mmdeploy/backend/ncnn/onnx2ncnn.exe')
    ncnn_ext_paths = glob(
        osp.join(mmdeploy_dir, 'mmdeploy/backend/ncnn/ncnn_ext.*'))
    for ncnn_ext_path in ncnn_ext_paths:
        os.remove(ncnn_ext_path)


def build_mmdeploy(cfg, mmdeploy_dir):

    args = [f'-D{k}={v}' for k, v in cfg.items()]

    # clear mmdeploy
    clear_mmdeploy(mmdeploy_dir)

    build_dir = osp.join(mmdeploy_dir, 'build')
    if not osp.exists(build_dir):
        os.mkdir(build_dir)

    # cmake cmd
    cmake_cmd = ' '.join(['cmake ..'] + args)

    # build cmd
    build_cmd = 'cmake --build . -- -j$(nproc) && cmake --install .'

    # build wheel
    bdist_cmd = 'python setup.py bdist_wheel'

    _call_command(cmake_cmd, build_dir)
    _call_command(build_cmd, build_dir)
    _call_command(bdist_cmd, mmdeploy_dir)


def get_dir_name(cfg, tag, default_name):
    if tag not in cfg:
        logging.warning(f'{tag} not found, use `{default_name}` as default.')
    else:
        default_name = cfg[tag]
        cfg = copy.deepcopy(cfg)
        cfg.pop(tag)
    return cfg, default_name


def create_package(cfg: Dict, mmdeploy_dir: str):
    build_dir = 'build'
    sdk_tar_name = 'sdk-tar'

    # load flags
    cfg, build_dir = get_dir_name(cfg, 'BUILD_NAME', build_dir)
    build_sdk_flag = cfg.get('MMDEPLOY_BUILD_SDK', False)
    if 'TAR_NAME' in cfg:
        cfg, sdk_tar_name = get_dir_name(cfg, 'TAR_NAME', sdk_tar_name)

    # create package directory.
    if osp.exists(build_dir):
        logging.info(f'{build_dir} existed, deleting...')
        shutil.rmtree(build_dir)
    os.mkdir(build_dir)

    logging.info(f'build mmdeploy in {build_dir}:')
    logging.debug(f'with config: {cfg}')

    try:
        # clear mmdeploy
        clear_mmdeploy(mmdeploy_dir)
        build_mmdeploy(cfg, mmdeploy_dir)
        _copy(osp.join(mmdeploy_dir, 'dist'), osp.join(build_dir, 'dist'))

        if build_sdk_flag:

            sdk_tar_dir = osp.join(build_dir, sdk_tar_name)

            # copy lib and install into sdk dir
            install_dir = osp.join(mmdeploy_dir, 'build/install/')
            _copy(install_dir, sdk_tar_dir)
            _remove_if_exist(osp.join(sdk_tar_dir, 'example', 'build'))

            # create sdk python api wheel
            python_api_lib_path = glob(
                osp.join(mmdeploy_dir, 'build/lib/mmdeploy_python.*.so'))
            num_libs = len(python_api_lib_path)
            assert num_libs == 1, f'Expect one api lib, but found {num_libs}.'
            python_api_lib_path = python_api_lib_path[0]

            sdk_python_package_dir = osp.join(build_dir, '.mmdeploy_python')
            _copy(PACKAGING_DIR, sdk_python_package_dir)
            _copy(python_api_lib_path,
                  osp.join(sdk_python_package_dir, 'mmdeploy_python'))
            create_wheel_cmd = 'python setup.py bdist_wheel'
            _call_command(create_wheel_cmd, sdk_python_package_dir)
            wheel_src_dir = osp.join(sdk_python_package_dir, 'dist')
            _copy(wheel_src_dir, osp.join(sdk_tar_dir, 'python'))

            # create tar file
            _create_tar(sdk_tar_dir, sdk_tar_dir + '.tar')

        logging.info('build finish.')

    except CalledProcessError:
        logging.error('build failed')
        exit()


def parse_args():
    parser = argparse.ArgumentParser(description='Build mmdeploy from yaml.')
    parser.add_argument('build_cfgs', help='The build config yaml file.')
    parser.add_argument('mmdeploy_dir', help='The source code of MMDeploy.')
    args = parser.parse_args()

    return args


def parse_configs(cfg_path: str):
    with open(cfg_path, mode='r') as f:
        cfgs = yaml.load(f, yaml.Loader)

    global_cfg = cfgs.get('global_config', dict())
    local_cfgs = cfgs.get('local_configs', [])

    if len(local_cfgs) == 0:
        merged_cfgs = [global_cfg]
    else:
        merged_cfgs = [copy.deepcopy(global_cfg) for _ in local_cfgs]

        for cfg, local_cfg in zip(merged_cfgs, local_cfgs):
            cfg.update(local_cfg)

    return merged_cfgs


def main():
    args = parse_args()
    cfgs = parse_configs(args.build_cfgs)
    mmdeploy_dir = osp.abspath(args.mmdeploy_dir)
    logging.info(f'Using mmdeploy_dir: {mmdeploy_dir}')

    logging.info(f'Using PACKAGING_DIR: {PACKAGING_DIR}')

    for cfg in cfgs:
        create_package(cfg, mmdeploy_dir)


if __name__ == '__main__':
    main()
