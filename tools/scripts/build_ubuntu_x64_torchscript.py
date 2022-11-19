# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import time

from ubuntu_utils import (cmd_result, cu_version_name, ensure_base_env,
                          get_job, pytorch_version)

g_jobs = 2


def install_libtorch(dep_dir):
    print('-' * 10 + 'install libtorch' + '-' * 10)
    time.sleep(2)

    os.chdir(dep_dir)
    unzipped_name = 'libtorch'
    if os.path.exists(unzipped_name):
        return os.path.join(dep_dir, unzipped_name)

    torch_version = pytorch_version()
    if torch_version is None:
        print('torch version is None, try 1.11.0')
        torch_version = '1.11.0'

    version_name = None

    # first check `nvcc` version, if failed, use `nvidia-smi`
    cuda = cmd_result(
        " nvcc --version | grep  release | awk '{print $5}' | awk -F , '{print $1}' "  # noqa: E501
    )
    if cuda is None or len(cuda) < 1:
        cuda = cmd_result(" nvidia-smi  | grep CUDA | awk '{print $9}' ")

    if cuda is not None and len(cuda) > 0:
        version_name = cu_version_name(cuda)
    else:
        version_name = 'cpu'

    filename = 'libtorch-shared-with-deps-{}%2B{}.zip'.format(
        torch_version, version_name)
    url = 'https://download.pytorch.org/libtorch/{}/{}'.format(
        version_name, filename)
    os.system('wget -q --show-progress {} -O libtorch.zip'.format(url))
    os.system('unzip libtorch.zip')
    if not os.path.exists(unzipped_name):
        print(
            'download or unzip libtorch from {} failed, please check https://pytorch.org/get-started/locally/'  # noqa: E501
            .format(url))
        return None
    return os.path.join(dep_dir, unzipped_name)


def install_mmdeploy(work_dir, libtorch_dir):
    print('-' * 10 + 'build and install mmdeploy' + '-' * 10)
    time.sleep(3)

    os.chdir(work_dir)
    os.system('git submodule init')
    os.system('git submodule update')

    if not os.path.exists('build'):
        os.system('mkdir build')

    os.system('rm -rf build/CMakeCache.txt')

    cmd = 'cd build &&  Torch_DIR={} cmake ..'.format(libtorch_dir)
    cmd += ' -DMMDEPLOY_BUILD_SDK=ON '
    cmd += ' -DMMDEPLOY_BUILD_EXAMPLES=ON '
    cmd += ' -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON '
    cmd += ' -DMMDEPLOY_TARGET_DEVICES=cpu '
    cmd += ' -DMMDEPLOY_TARGET_BACKENDS=torchscript '
    cmd += ' -DTORCHSCRIPT_DIR={} '.format(libtorch_dir)
    os.system(cmd)

    os.system('cd build && make -j {} && make install'.format(g_jobs))
    os.system('python3 -m pip install -e .')
    try:
        import mmcv
        print(mmcv.__version__)
        os.system('python3 tools/check_env.py')
    except Exception:
        print('Please install torch & mmcv later.. ≥▽≤')
    return 0


def main():
    """Auto install mmdeploy with ort. To verify this script:

    1) use `sudo docker run -v /path/to/mmdeploy:/root/mmdeploy -v /path/to/Miniconda3-latest-Linux-x86_64.sh:/root/miniconda.sh -it ubuntu:18.04 /bin/bash` # noqa: E501
    2) install conda and setup python environment
    3) run `python3 tools/scripts/build_ubuntu_x64_torchscript.py`

    Returns:
        _type_: _description_
    """
    global g_jobs
    g_jobs = get_job(sys.argv)
    print('g_jobs {}'.format(g_jobs))

    work_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
    dep_dir = os.path.abspath(os.path.join(work_dir, '..', 'mmdeploy-dep'))
    if not os.path.exists(dep_dir):
        if os.path.isfile(dep_dir):
            print('{} already exists and it is a file, exit.'.format(work_dir))
            return -1
        os.mkdir(dep_dir)

    success = ensure_base_env(work_dir, dep_dir)
    if success != 0:
        return -1

    libtorch_dir = install_libtorch(dep_dir)

    if libtorch_dir is None:
        return -1

    if install_mmdeploy(work_dir, libtorch_dir) != 0:
        return -1

    if os.path.exists('~/mmdeploy.env'):
        print('Please source ~/mmdeploy.env to setup your env !')
        os.system('cat ~/mmdeploy.env')


if __name__ == '__main__':
    main()
