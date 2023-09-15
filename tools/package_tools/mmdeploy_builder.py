# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import logging
import os
import os.path as osp
import platform
import re
import shutil
import sys
from glob import glob
from subprocess import check_output, run
from typing import Dict

import yaml
from packaging import version

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CUR_DIR = osp.dirname(osp.abspath(__file__))
MMDEPLOY_DIR = osp.abspath(osp.join(CUR_DIR, '../..'))
PACKAGING_DIR = osp.join(CUR_DIR, 'packaging')
VERSION_FILE = osp.join(MMDEPLOY_DIR, 'mmdeploy', 'version.py')


def get_version(version_file):
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def _remove_if_exist(path):
    if osp.exists(path):
        logging.info(f'Remove path: {path}')
        if osp.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    if osp.islink(path):
        os.remove(path)


def _copy(src_path, dst_path):
    assert osp.exists(src_path), f'src path: {src_path} not exist.'

    logging.info(f'copy path: {src_path} to {dst_path}.')
    if osp.isdir(src_path):
        if osp.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path, symlinks=True)
    else:
        shutil.copy(src_path, dst_path)


def _call_command(cmd, cwd, stdout=None, stderr=None):
    if cmd == '':
        return
    logging.info(f'Process cmd: {cmd}')
    logging.info(f'work_path: {cwd}')
    try:
        ret = run(cmd, stdout=stdout, stderr=stderr, cwd=cwd, shell=True)
        if ret.returncode != 0:
            logging.error(f'Process cmd: "{cmd}"'
                          f' failed with returncode: {ret.returncode}')
            exit(-1)
    except Exception:
        logging.error(f'Process cmd: {cmd} failed.')
        exit(-1)


def _create_bdist_cmd(cfg, c_ext=False, dist_dir=None):

    bdist_tags = cfg.get('bdist_tags', {})

    # base
    bdist_cmd = 'python setup.py bdist_wheel '

    # platform
    bdist_cmd += f' --plat-name {cfg["PLATFORM_TAG"]} '

    # python tag
    python_tag = f'cp{sys.version_info.major}{sys.version_info.minor}'\
        if c_ext else 'py3'
    if 'python_tag' in bdist_tags:
        python_tag = bdist_tags['python_tag']
    bdist_cmd += f' --python-tag {python_tag} '

    # dist dir
    if dist_dir is not None:
        dist_dir = osp.abspath(dist_dir)
        bdist_cmd += f' --dist-dir {dist_dir} '
    return bdist_cmd


def clear_mmdeploy():
    logging.info(f'Cleaning mmdeploy: {MMDEPLOY_DIR}')

    def _remove_in_mmdeploy(path):
        remove_dir = osp.join(MMDEPLOY_DIR, path)
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
    _remove_in_mmdeploy('mmdeploy/backend/ncnn/mmdeploy_onnx2ncnn')
    _remove_in_mmdeploy('mmdeploy/backend/ncnn/mmdeploy_onnx2ncnn.exe')
    ncnn_ext_paths = glob(
        osp.join(MMDEPLOY_DIR, 'mmdeploy/backend/ncnn/ncnn_ext.*'))
    for ncnn_ext_path in ncnn_ext_paths:
        os.remove(ncnn_ext_path)

    # remove ts_optmizer
    ts_optimizer_paths = glob(
        osp.join(MMDEPLOY_DIR, 'mmdeploy/backend/torchscript/ts_optimizer.*'))
    for ts_optimizer_path in ts_optimizer_paths:
        os.remove(ts_optimizer_path)


def check_env(cfg: Dict):
    env_info = {}

    cmake_envs = cfg.get('cmake_cfg', dict())

    # system
    platform_system = platform.system().lower()
    platform_machine = platform.machine().lower()
    env_info['system'] = platform_system
    env_info['machine'] = platform_machine

    # CUDA version
    cuda_version = 'unknown'

    CUDA_TOOLKIT_ROOT_DIR = os.environ.get('CUDA_TOOLKIT_ROOT_DIR', '')
    CUDA_TOOLKIT_ROOT_DIR = cmake_envs.get('CUDA_TOOLKIT_ROOT_DIR',
                                           CUDA_TOOLKIT_ROOT_DIR)
    CUDA_TOOLKIT_ROOT_DIR = osp.expandvars(CUDA_TOOLKIT_ROOT_DIR)
    nvcc_cmd = ('nvcc' if len(CUDA_TOOLKIT_ROOT_DIR) <= 0 else osp.join(
        CUDA_TOOLKIT_ROOT_DIR, 'bin', 'nvcc'))

    try:
        nvcc = check_output(f'"{nvcc_cmd}" -V', shell=True)
        nvcc = nvcc.decode('utf-8').strip()
        pattern = r'Cuda compilation tools, release (\d+.\d+)'
        match = re.search(pattern, nvcc)
        if match is not None:
            cuda_version = match.group(1)
    except Exception:
        pass

    env_info['cuda_v'] = cuda_version

    # ONNX Runtime version
    onnxruntime_version = 'unknown'

    ONNXRUNTIME_DIR = os.getenv('ONNXRUNTIME_DIR', '')
    ONNXRUNTIME_DIR = cmake_envs.get('ONNXRUNTIME_DIR', ONNXRUNTIME_DIR)
    ONNXRUNTIME_DIR = osp.expandvars(ONNXRUNTIME_DIR)

    if osp.exists(ONNXRUNTIME_DIR):
        with open(osp.join(ONNXRUNTIME_DIR, 'VERSION_NUMBER'), mode='r') as f:
            onnxruntime_version = f.readlines()[0].strip()

    env_info['ort_v'] = onnxruntime_version

    # TensorRT version
    tensorrt_version = 'unknown'

    TENSORRT_DIR = os.getenv('TENSORRT_DIR', '')
    TENSORRT_DIR = cmake_envs.get('TENSORRT_DIR', TENSORRT_DIR)
    TENSORRT_DIR = osp.expandvars(TENSORRT_DIR)

    if osp.exists(TENSORRT_DIR):
        with open(
                osp.join(TENSORRT_DIR, 'include', 'NvInferVersion.h'),
                mode='r') as f:
            data = f.read()
            major = re.search(r'#define NV_TENSORRT_MAJOR (\d+)', data)
            minor = re.search(r'#define NV_TENSORRT_MINOR (\d+)', data)
            patch = re.search(r'#define NV_TENSORRT_PATCH (\d+)', data)
            build = re.search(r'#define NV_TENSORRT_BUILD (\d+)', data)
            if major is not None and minor is not None and patch is not None \
                    and build is not None:
                tensorrt_version = (f'{major.group(1)}.' +
                                    f'{minor.group(1)}.' +
                                    f'{patch.group(1)}.' + f'{build.group(1)}')

    env_info['trt_v'] = tensorrt_version

    return env_info


def build_mmdeploy(cfg: Dict):
    build_dir = osp.join(MMDEPLOY_DIR, 'build')
    if not osp.exists(build_dir):
        os.mkdir(build_dir)

    cmake_cfg = cfg['cmake_cfg']
    cmake_options = [f'-D{k}="{v}"' for k, v in cmake_cfg.items() if v != '']
    if sys.platform == 'win32':
        cmake_windows_options = '-A x64 -T v142'
        if 'CUDA_PATH' in os.environ:
            cmake_windows_options += ',cuda="%CUDA_PATH%"'
        cmake_options = [cmake_windows_options] + cmake_options

    # configure
    cmake_cmd = ' '.join(['cmake ..'] + cmake_options)
    _call_command(cmake_cmd, build_dir)
    # build
    if sys.platform == 'win32':
        build_cmd = 'cmake --build . --config Release -- /m'
    else:
        build_cmd = 'cmake --build . -- -j$(nproc)'
    _call_command(build_cmd, build_dir)
    # install
    install_cmd = 'cmake --install . --config Release'
    _call_command(install_cmd, build_dir)


def copy_thirdparty(cfg: Dict, sdk_path: str):
    thirdparty_dir = osp.join(sdk_path, 'thirdparty')
    os.mkdir(thirdparty_dir)

    def _copy_needed(src_dir, dst_dir, needed):
        if not osp.exists(dst_dir):
            os.makedirs(dst_dir)
        for path in needed:
            src_path = osp.join(src_dir, path[0])
            dst_path = osp.join(dst_dir, path[0])
            _copy(src_path, dst_path)
            if len(path) == 1 or path[1] == '**':
                continue

            old_dir = os.getcwd()
            os.chdir(dst_path)
            files = glob('**', recursive=True)
            reserve = []
            for pattern in path[1:]:
                reserve.extend(glob(pattern, recursive=True))

            for file in files:
                if file not in reserve:
                    _remove_if_exist(file)
            os.chdir(old_dir)

    # copy onnxruntime, tensorrt
    backend = cfg['cmake_cfg']['MMDEPLOY_TARGET_BACKENDS']
    if 'ort' in backend:
        src_dir = cfg['cmake_cfg']['ONNXRUNTIME_DIR']
        dst_dir = osp.join(thirdparty_dir, 'onnxruntime')
        needed = [('include', '**'), ('lib', '**')]
        _copy_needed(src_dir, dst_dir, needed)
    if 'trt' in backend:
        src_dir = cfg['cmake_cfg']['TENSORRT_DIR']
        dst_dir = osp.join(thirdparty_dir, 'tensorrt')
        needed = [('include', '**'),
                  ('lib', 'libnvinfer_builder_resource.so*', 'libnvinfer.so*',
                   'libnvinfer_plugin.so*', 'nvinfer_builder_resource.*',
                   'nvinfer*', 'nvinfer_plugin*')]
        _copy_needed(src_dir, dst_dir, needed)


def copy_scripts(sdk_path: str):
    scripts_base = osp.join(MMDEPLOY_DIR, 'tools', 'package_tools', 'scripts')
    if sys.platform == 'win32':
        src_dir = osp.join(scripts_base, 'windows')
    elif sys.platform == 'linux':
        src_dir = osp.join(scripts_base, 'linux')
    else:
        raise Exception('unsupported')
    files = glob(osp.join(src_dir, '*'))
    for file in files:
        filename = osp.basename(file)
        src_path = osp.join(src_dir, filename)
        dst_path = osp.join(sdk_path, filename)
        _copy(src_path, dst_path)


def copy_onnxruntime(cfg, dst_dir):
    ort_root = cfg['cmake_cfg']['ONNXRUNTIME_DIR']
    patterns = ['libonnxruntime.so.*', 'onnxruntime.dll']
    for pattern in patterns:
        src_lib = glob(osp.join(ort_root, 'lib', pattern))
        if len(src_lib) > 0:
            dst_lib = osp.join(dst_dir, osp.basename(src_lib[0]))
            _copy(src_lib[0], dst_lib)


def create_mmdeploy(cfg: Dict, work_dir: str):
    if cfg['BUILD_MMDEPLOY'] == 'OFF':
        logging.info('Skip build mmdeploy package')
        return

    dist_dir = osp.join(work_dir, 'mmdeploy')
    if osp.exists(dist_dir):
        logging.info('mmdeploy existed, deleting...')
        shutil.rmtree(dist_dir)

    clear_mmdeploy()
    build_mmdeploy(cfg)

    # copy libonnxruntime.so.x.y.z
    backend = cfg['cmake_cfg']['MMDEPLOY_TARGET_BACKENDS']
    if 'ort' in backend:
        dst_dir = osp.join(MMDEPLOY_DIR, 'mmdeploy', 'lib')
        copy_onnxruntime(cfg, dst_dir)

    # build wheel
    build_dir = osp.join(MMDEPLOY_DIR, 'build')
    _remove_if_exist(osp.join(build_dir, 'lib'))
    _remove_if_exist(osp.join(build_dir, 'lib', 'Release'))
    bdist_cmd = _create_bdist_cmd(cfg, c_ext=False, dist_dir=dist_dir)
    _call_command(bdist_cmd, MMDEPLOY_DIR)


def create_mmdeploy_runtime(cfg: Dict, work_dir: str):
    cmake_cfg = cfg['cmake_cfg']
    if cmake_cfg['MMDEPLOY_BUILD_SDK'] == 'OFF' or \
            cmake_cfg['MMDEPLOY_BUILD_SDK_PYTHON_API'] == 'OFF':
        logging.info('Skip build mmdeploy sdk python api')
        return

    for python_version in ['3.6', '3.7', '3.8', '3.9', '3.10', '3.11']:
        _version = version.parse(python_version)
        python_major = _version.major
        python_minor = _version.minor
        # create sdk python api wheel
        sdk_python_package_dir = osp.join(work_dir, '.mmdeploy_runtime')
        _copy(PACKAGING_DIR, sdk_python_package_dir)
        _copy(
            VERSION_FILE,
            osp.join(sdk_python_package_dir, 'mmdeploy_runtime', 'version.py'),
        )

        # build mmdeploy_runtime
        python_executable = shutil.which('python')\
            .replace('mmdeploy-3.6', f'mmdeploy-{python_version}')
        cmake_options = [
            f'-D{k}="{v}"' for k, v in cmake_cfg.items() if v != ''
        ]
        cmake_options.append(
            f'-DMMDeploy_DIR={MMDEPLOY_DIR}/build/install/lib/cmake/MMDeploy')
        cmake_options.append(f'-DPYTHON_EXECUTABLE={python_executable}')
        if sys.platform == 'win32':
            cmake_options.append('-A x64 -T v142')
            if 'CUDA_PATH' in os.environ:
                cmake_options[-1] += ',cuda="%CUDA_PATH%"'
        cmake_cmd = ' '.join(['cmake ../csrc/mmdeploy/apis/python'] +
                             cmake_options)
        build_dir = osp.join(MMDEPLOY_DIR, 'build_python')
        _remove_if_exist(build_dir)
        os.mkdir(build_dir)
        _call_command(cmake_cmd, build_dir)
        if sys.platform == 'win32':
            build_cmd = 'cmake --build . --config Release -- /m'
        else:
            build_cmd = 'cmake --build . -- -j$(nproc)'
        _call_command(build_cmd, build_dir)

        # copy api lib
        python_api_lib_path = []
        lib_patterns = ['*mmdeploy_runtime*.so', '*mmdeploy_runtime*.pyd']
        for pattern in lib_patterns:
            python_api_lib_path.extend(
                glob(
                    osp.join(MMDEPLOY_DIR, 'build_python/**', pattern),
                    recursive=True,
                ))
        _copy(
            python_api_lib_path[0],
            osp.join(sdk_python_package_dir, 'mmdeploy_runtime'),
        )
        _remove_if_exist(osp.join(MMDEPLOY_DIR, 'build_python'))

        # copy net & mmdeploy
        if sys.platform == 'win32':
            libs_to_copy = ['*net.dll', 'mmdeploy.dll']
            search_dir = osp.join(MMDEPLOY_DIR, 'build', 'install', 'bin')
        elif sys.platform == 'linux':
            mmdeploy_version = get_version(VERSION_FILE)
            mmdeploy_version = version.parse(mmdeploy_version)
            libs_to_copy = [
                '*net.so', f'*mmdeploy.so.{mmdeploy_version.major}'
            ]
            search_dir = osp.join(MMDEPLOY_DIR, 'build', 'install', 'lib')
        else:
            raise Exception('unsupported')

        for pattern in libs_to_copy:
            files = glob(osp.join(search_dir, pattern))
            for file in files:
                _copy(file, osp.join(sdk_python_package_dir,
                                     'mmdeploy_runtime'))

        # copy onnxruntime
        if 'ort' in cfg['cmake_cfg']['MMDEPLOY_TARGET_BACKENDS']:
            copy_onnxruntime(
                cfg, osp.join(sdk_python_package_dir, 'mmdeploy_runtime'))

        # bdist
        sdk_wheel_dir = osp.join(work_dir, 'mmdeploy_runtime')
        cfg['bdist_tags'] = {'python_tag': f'cp{python_major}{python_minor}'}
        bdist_cmd = _create_bdist_cmd(cfg, c_ext=True, dist_dir=sdk_wheel_dir)
        if 'cuda' in cmake_cfg['MMDEPLOY_TARGET_DEVICES']:
            bdist_cmd += ' --use-gpu'
        _call_command(bdist_cmd, sdk_python_package_dir)
        _remove_if_exist(sdk_python_package_dir)


def create_sdk(cfg: Dict, work_dir: str):
    cmake_cfg = cfg['cmake_cfg']
    if cmake_cfg['MMDEPLOY_BUILD_SDK'] == 'OFF':
        logging.info('Skip build mmdeploy sdk')
        return

    cfg = copy.deepcopy(cfg)
    cfg['cmake_cfg']['MMDEPLOY_BUILD_SDK_PYTHON_API'] = 'OFF'
    clear_mmdeploy()
    build_mmdeploy(cfg)

    sdk_root = osp.abspath(osp.join(work_dir, 'sdk'))
    build_sdk_name = cfg['BUILD_SDK_NAME']
    env_info = check_env(cfg)
    mmdeploy_version = get_version(VERSION_FILE)
    build_sdk_name = build_sdk_name.format(
        mmdeploy_v=mmdeploy_version, **env_info)
    sdk_path = osp.join(sdk_root, build_sdk_name)

    if osp.exists(sdk_path):
        logging.info(f'{sdk_path}, deleting...')
        shutil.rmtree(sdk_path)
    os.makedirs(sdk_path)

    install_dir = osp.join(MMDEPLOY_DIR, 'build/install/')
    _copy(install_dir, sdk_path)
    _copy(f'{MMDEPLOY_DIR}/demo/python', f'{sdk_path}/example/python')
    _remove_if_exist(osp.join(sdk_path, 'example', 'build'))

    # copy thirdparty
    copy_thirdparty(cfg, sdk_path)
    # copy scripts
    copy_scripts(sdk_path)


def create_package(cfg: Dict, work_dir: str):
    create_mmdeploy(cfg, work_dir)
    create_sdk(cfg, work_dir)
    create_mmdeploy_runtime(cfg, work_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Build mmdeploy from yaml.')
    parser.add_argument('--config', help='The build config yaml file.')
    parser.add_argument(
        '--output-dir', default='.', help='Output package directory.')
    args = parser.parse_args()

    return args


def parse_configs(cfg_path: str):
    with open(cfg_path, mode='r') as f:
        config = yaml.load(f, yaml.Loader)
    logging.info(f'Load config\n{yaml.dump(config)}')
    return config


def main():
    args = parse_args()
    cfg = parse_configs(args.config)
    work_dir = osp.abspath(args.output_dir)
    logging.info(f'Using mmdeploy_dir: {MMDEPLOY_DIR}')
    logging.info(f'Using output_dir: {work_dir}')
    create_package(cfg, work_dir)


if __name__ == '__main__':
    main()
