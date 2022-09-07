# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import time

from ubuntu_utils import cmd_result, ensure_base_env, get_job

g_jobs = 2


def install_protobuf(dep_dir) -> int:
    """build and install protobuf.

    Args:
        wor_dir (_type_): _description_

    Returns:
        : _description_
    """
    print('-' * 10 + 'install protobuf' + '-' * 10)

    os.chdir(dep_dir)
    if not os.path.exists('protobuf-3.20.0'):
        os.system(
            'wget https://github.com/protocolbuffers/protobuf/releases/download/v3.20.0/protobuf-cpp-3.20.0.tar.gz'  # noqa: E501
        )
        os.system('tar xvf protobuf-cpp-3.20.0.tar.gz')

    os.chdir(os.path.join(dep_dir, 'protobuf-3.20.0'))

    install_dir = os.path.join(dep_dir, 'pbinstall')
    os.system('./configure --prefix={}'.format(install_dir))
    os.system('make -j {} && make install'.format(g_jobs))
    protoc = os.path.join(dep_dir, 'pbinstall', 'bin', 'protoc')

    print('protoc \t:{}'.format(cmd_result('{} --version'.format(protoc))))
    return 0


def install_pyncnn(dep_dir):
    print('-' * 10 + 'build and install pyncnn' + '-' * 10)
    time.sleep(2)

    # generate unzip and build dir
    os.chdir(dep_dir)

    # git clone
    if not os.path.exists('ncnn'):
        os.system(
            'git clone --depth 1 --branch 20220729  https://github.com/tencent/ncnn && cd ncnn'  # noqa: E501
        )

    ncnn_dir = os.path.join(dep_dir, 'ncnn')
    os.chdir(ncnn_dir)

    # update submodule pybind11, gslang not required
    os.system('git submodule init && git submodule update python/pybind11')
    # build
    if not os.path.exists('build'):
        os.system('mkdir build')

    os.chdir(os.path.join(ncnn_dir, 'build'))
    pb_install = os.path.join(dep_dir, 'pbinstall')
    pb_bin = os.path.join(pb_install, 'bin', 'protoc')
    pb_lib = os.path.join(pb_install, 'lib', 'libprotobuf.so')
    pb_include = os.path.join(pb_install, 'include')

    cmd = 'cmake .. '
    cmd += ' -DNCNN_PYTHON=ON '
    cmd += ' -DProtobuf_LIBRARIES={} '.format(pb_lib)
    cmd += ' -DProtobuf_PROTOC_EXECUTABLE={} '.format(pb_bin)
    cmd += ' -DProtobuf_INCLUDE_DIR={} '.format(pb_include)
    cmd += ' && make -j {} '.format(g_jobs)
    cmd += ' && make install '
    os.system(cmd)

    # install
    os.chdir(ncnn_dir)
    os.system('cd python && python -m pip install -e .')
    ncnn_cmake_dir = os.path.join(ncnn_dir, 'build', 'install', 'lib', 'cmake',
                                  'ncnn')
    assert (os.path.exists(ncnn_cmake_dir))
    print('ncnn cmake dir \t:{}'.format(ncnn_cmake_dir))
    print('\n')
    return ncnn_cmake_dir


def install_mmdeploy(work_dir, dep_dir, ncnn_cmake_dir):
    print('-' * 10 + 'build and install mmdeploy' + '-' * 10)
    time.sleep(3)

    os.chdir(work_dir)
    os.system('git submodule init')
    os.system('git submodule update')

    if not os.path.exists('build'):
        os.system('mkdir build')

    pb_install = os.path.join(dep_dir, 'pbinstall')
    pb_bin = os.path.join(pb_install, 'bin', 'protoc')
    pb_lib = os.path.join(pb_install, 'lib', 'libprotobuf.so')
    pb_include = os.path.join(pb_install, 'include')

    cmd = 'cd build && cmake ..'
    cmd += ' -DCMAKE_C_COMPILER=gcc-7 '
    cmd += ' -DCMAKE_CXX_COMPILER=g++-7 '
    cmd += ' -DMMDEPLOY_BUILD_SDK=ON '
    cmd += ' -DMMDEPLOY_BUILD_EXAMPLES=ON '
    cmd += ' -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON '
    cmd += ' -DMMDEPLOY_TARGET_DEVICES=cpu '
    cmd += ' -DMMDEPLOY_TARGET_BACKENDS=ncnn '
    cmd += ' -DProtobuf_PROTOC_EXECUTABLE={} '.format(pb_bin)
    cmd += ' -DProtobuf_LIBRARIES={} '.format(pb_lib)
    cmd += ' -DProtobuf_INCLUDE_DIR={} '.format(pb_include)
    cmd += ' -Dncnn_DIR={} '.format(ncnn_cmake_dir)
    os.system(cmd)

    os.system('cd build && make -j {} && make install'.format(g_jobs))
    os.system('python3 -m pip install -v -e .')
    return 0


def main():
    """Auto install mmdeploy with ncnn. To verify this script:

    1) use `sudo docker run -v /path/to/mmdeploy:/root/mmdeploy -v /path/to/Miniconda3-latest-Linux-x86_64.sh:/root/miniconda.sh -it ubuntu:18.04 /bin/bash` # noqa: E501
    2) install conda and setup python environment
    3) run `python3 tools/scripts/build_ubuntu_x64_ncnn.py`

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

    if install_protobuf(dep_dir) != 0:
        return -1

    ncnn_cmake_dir = install_pyncnn(dep_dir)

    if install_mmdeploy(work_dir, dep_dir, ncnn_cmake_dir) != 0:
        return -1

    if len(envs) > 0:
        print(
            'We recommend that you set the following environment variables:\n')
        for env in envs:
            print(env)
            print('\n')


if __name__ == '__main__':
    main()
