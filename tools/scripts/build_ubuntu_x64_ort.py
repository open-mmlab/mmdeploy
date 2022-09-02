# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import time

from ubuntu_utils import ensure_base_env, get_job

g_jobs = 2


def install_ort(dep_dir):
    print('-' * 10 + 'install ort' + '-' * 10)
    time.sleep(2)

    # generate unzip and build dir
    os.chdir(dep_dir)

    # install python onnxruntime
    os.system('python3 -m pip install onnxruntime==1.8.1')
    # git clone
    if not os.path.exists('onnxruntime-linux-x64-1.8.1'):
        os.system(
            'wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz'  # noqa: E501
        )
        os.system('tar xvf  onnxruntime-linux-x64-1.8.1.tgz')

    ort_dir = os.path.join(dep_dir, 'onnxruntime-linux-x64-1.8.1')
    print('onnxruntime dir \t:{}'.format(ort_dir))
    print('\n')
    return ort_dir


def install_mmdeploy(work_dir, ort_dir):
    print('-' * 10 + 'build and install mmdeploy' + '-' * 10)
    time.sleep(3)

    os.chdir(work_dir)
    os.system('git submodule init')
    os.system('git submodule update')

    if not os.path.exists('build'):
        os.system('mkdir build')

    cmd = 'cd build && cmake ..'
    cmd += ' -DCMAKE_C_COMPILER=gcc-7 '
    cmd += ' -DCMAKE_CXX_COMPILER=g++-7 '
    cmd += ' -DMMDEPLOY_BUILD_SDK=ON '
    cmd += ' -DMMDEPLOY_BUILD_EXAMPLES=ON '
    cmd += ' -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON '
    cmd += ' -DMMDEPLOY_TARGET_DEVICES=cpu '
    cmd += ' -DMMDEPLOY_TARGET_BACKENDS=ort '
    cmd += ' -DONNXRUNTIME_DIR={} '.format(ort_dir)
    os.system(cmd)

    os.system('cd build && make -j {} && make install'.format(g_jobs))
    os.system('python3 -m pip install -e .')
    return 0


def main():
    """Auto install mmdeploy with ort. To verify this script:

    1) use `sudo docker run -v /path/to/mmdeploy:/root/mmdeploy -v /path/to/Miniconda3-latest-Linux-x86_64.sh:/root/miniconda.sh -it ubuntu:18.04 /bin/bash` # noqa: E501
    2) install conda and setup python environment
    3) run `python3 tools/scripts/build_ubuntu_x64_ort.py`

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

    success, envs = ensure_base_env(work_dir, dep_dir)
    if success != 0:
        return -1

    ort_dir = install_ort(dep_dir)

    if install_mmdeploy(work_dir, ort_dir) != 0:
        return -1

    if len(envs) > 0:
        print(
            'We recommend that you set the following environment variables:\n')
        for env in envs:
            print(env)
            print('\n')


if __name__ == '__main__':
    main()
