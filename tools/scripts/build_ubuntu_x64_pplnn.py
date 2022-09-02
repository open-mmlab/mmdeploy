# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import time

from ubuntu_utils import cmd_result, ensure_base_env, get_job

g_jobs = 2


def install_pplcv(dep_dir, build_cuda):
    print('-' * 10 + 'install pplcv' + '-' * 10)
    time.sleep(2)

    os.chdir(dep_dir)

    pplcv_dir = os.path.join(dep_dir, 'ppl.cv')

    # git clone
    if not os.path.exists(pplcv_dir):
        os.system(
            'git clone --depth 1 --branch v0.7.0 https://github.com/openppl-public/ppl.cv/'  # noqa: E501
        )

    # build
    os.chdir(pplcv_dir)
    if build_cuda is True:
        os.system('./build.sh cuda')
        pplcv_cmake_dir = os.path.join(pplcv_dir,
                                       'cuda-build/install/lib/cmake/ppl')
    else:
        os.system('./build.sh x86_64')
        pplcv_cmake_dir = os.path.join(pplcv_dir,
                                       'x86-64-build/install/lib/cmake/ppl')

    print('\n')
    return pplcv_cmake_dir


def install_pplnn(dep_dir, build_cuda):
    print('-' * 10 + 'install pplnn' + '-' * 10)
    time.sleep(2)

    # generate unzip and build dir
    os.chdir(dep_dir)

    pplnn_dir = os.path.join(dep_dir, 'ppl.nn')

    # git clone
    if not os.path.exists(pplnn_dir):
        os.system(
            'git clone --depth 1 --branch v0.8.2 https://github.com/openppl-public/ppl.nn/'  # noqa: E501
        )

    # build
    os.chdir(pplnn_dir)
    if build_cuda is True:
        os.system(
            './build.sh -DPPLNN_USE_CUDA=ON -DPPLNN_USE_X86_64=ON  -DPPLNN_ENABLE_PYTHON_API=ON'  # noqa: E501
        )
    else:
        os.system(
            './build.sh -DPPLNN_USE_X86_64=ON  -DPPLNN_ENABLE_PYTHON_API=ON'  # noqa: E501
        )

    os.system('cd python/package && ./build.sh')
    os.system(
        'cd /tmp/pyppl-package/dist && python3 -m pip install pyppl*.whl --force-reinstall'  # noqa: E501
    )

    pplnn_cmake_dir = os.path.join(pplnn_dir,
                                   'pplnn-build/install/lib/cmake/ppl')
    print('\n')
    return pplnn_cmake_dir


def install_mmdeploy(work_dir, pplnn_cmake_dir, pplcv_cmake_dir, build_cuda):
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
    cmd += ' -DMMDEPLOY_TARGET_BACKENDS=pplnn '

    if build_cuda is True:
        cmd += ' -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" '
    else:
        cmd += ' -DMMDEPLOY_TARGET_DEVICES=cpu '

    cmd += ' -Dpplcv_DIR={} '.format(pplcv_cmake_dir)
    cmd += ' -Dpplnn_DIR={} '.format(pplnn_cmake_dir)
    os.system(cmd)

    os.system('cd build && make -j {} && make install'.format(g_jobs))
    os.system('python3 -m pip install -e .')
    return 0


def main():
    """Auto install mmdeploy with pplnn. To verify this script:

    1) use `sudo docker run -v /path/to/mmdeploy:/root/mmdeploy -v /path/to/Miniconda3-latest-Linux-x86_64.sh:/root/miniconda.sh -it ubuntu:18.04 /bin/bash` # noqa: E501
    2) install conda and setup python environment
    3) run `python3 tools/scripts/build_ubuntu_x64_pplnn.py`

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

    # enable g++ and gcc
    gplus = cmd_result('which g++')
    if gplus is None or len(gplus) < 1:
        sudo = 'sudo'
        if 'root' in cmd_result('whoami'):
            sudo = ''
        os.system(
            '{} update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 200'  # noqa: E501
            .format(sudo))
        os.system(
            '{} update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 200'  # noqa: E501
            .format(sudo))

    # install pplcv and pplnn
    nvcc = cmd_result('which nvcc')
    build_cuda = False
    if nvcc is not None and len(nvcc) > 1:
        build_cuda = True
    pplcv_cmake_dir = install_pplcv(dep_dir, build_cuda)
    pplnn_cmake_dir = install_pplnn(dep_dir, build_cuda)
    if install_mmdeploy(work_dir, pplnn_cmake_dir, pplcv_cmake_dir,
                        build_cuda) != 0:
        return -1

    if len(envs) > 0:
        print(
            'We recommend that you set the following environment variables:\n')
        for env in envs:
            print(env)
            print('\n')


if __name__ == '__main__':
    main()
