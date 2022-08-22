# Copyright (c) OpenMMLab. All rights reserved.
import os
import time

from ubuntu_utils import (cmd_result, cu_version_name, simple_check_install,
                          version_major, version_minor)

g_jobs = 2


def install_protobuf(dep_dir) -> int:
    """build and install protobuf.

    Args:
        wor_dir (_type_): _description_

    Returns:
        : _description_
    """
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


def ensure_env(work_dir, dep_dir):
    """check python, cmake and torch environment.

    Returns:
        int: _description_
    """
    envs = []
    print('-' * 10 + 'ensure env' + '-' * 10)

    os.system('python3 -m ensurepip')

    sudo = 'sudo'
    if 'root' in cmd_result('whoami'):
        sudo = ''

    # check ubuntu
    ubuntu = cmd_result(
        """ lsb_release -a 2>/dev/null | grep "Release" | tail -n 1 | awk '{print $NF}' """  # noqa: E501
    )

    # check cmake version
    cmake = cmd_result('which cmake')
    if cmake is None or len(cmake) < 1:
        print('cmake not found, try install cmake ..', end='')
        os.system('python3 -m pip install cmake>=3.14.0')

        cmake = cmd_result('which cmake')
        if cmake is None or len(cmake) < 1:
            env = 'export PATH=${PATH}:~/.local/bin'
            os.system(env)
            envs.append(env)

            cmake = cmd_result('which cmake')
            if cmake is None or len(cmake) < 1:
                print('Check cmake failed.')
                return -1, envs
        print('success')

    # check  make
    make = cmd_result('which make')
    if make is None or len(make) < 1:
        print('make not found, try install make ..', end='')
        os.system('{} apt update --fix-missing'.format(sudo))

        os.system(
            '{} DEBIAN_FRONTEND="noninteractive"  apt install make'.format(
                sudo))
        make = cmd_result('which make')
        if make is None or len(make) < 1:
            print('Check make failed.')
            return -1, envs
        print('success')

    # check g++ version
    gplus = cmd_result('which g++-7')
    if gplus is None or len(gplus) < 1:
        # install g++
        print('g++-7 not found, try install g++-7 ..', end='')
        os.system(
            '{} DEBIAN_FRONTEND="noninteractive" apt install software-properties-common -y'  # noqa: E501
            .format(sudo))  # noqa: E501
        os.system('{} apt update'.format(sudo))
        if ubuntu is None or len(ubuntu) < 1 or version_major(ubuntu) <= 18:
            os.system(
                '{} add-apt-repository ppa:ubuntu-toolchain-r/test -y'.format(
                    sudo))
        os.system('{} apt install gcc-7 g++-7 -y'.format(sudo))

        gplus = cmd_result('which g++-7')
        if gplus is None or len(gplus) < 1:
            print('Check g++-7 failed.')
            return -1, envs
        os.system(
            '{} update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 200'  # noqa: E501
            .format(sudo))
        os.system(
            '{} update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 200'  # noqa: E501
            .format(sudo))
        print('success')

    # wget
    wget = simple_check_install('wget', sudo)

    # check torch and mmcv, we try to install mmcv, it is not compulsory
    mmcv_version = None
    torch_version = None
    try:
        import torch
        torch_version = torch.__version__

        try:
            import mmcv
            mmcv_version = mmcv.__version__
        except Exception:
            # install mmcv
            print('mmcv not found, try install mmcv ..', end='')
            cuda_version = cmd_result(
                " nvidia-smi  | grep CUDA | awk '{print $9}' ")
            if cuda_version is not None and len(cuda_version) > 2:

                format_version = str(version_major(torch_version)) + '.' + str(
                    version_minor(torch_version)) + '.0'
                mmcv_url = 'https://download.openmmlab.com/mmcv/dist/{}/torch{}/index.html'.format(  # noqa: E501
                    cu_version_name(cuda_version), format_version)
                http_ret = cmd_result('wget {}'.format(mmcv_url))
                if '404' not in http_ret:
                    mmcv_version = '1.5.0'
                    cmd = 'python3 -m pip install mmcv-full={} -f {}'.format(
                        mmcv_version, mmcv_url)
                    os.system(cmd)
                print('success')
    except Exception:
        pass

    # git
    git = simple_check_install('git', sudo)

    # protoc
    install_protobuf(dep_dir)
    protoc = os.path.join(dep_dir, 'pbinstall', 'bin', 'protoc')

    # opencv
    ocv = cmd_result('which opencv_version')
    if ocv is None or len(ocv) < 1:
        print('ocv not found, try install git ..', end='')
        os.system(
            '{} add-apt-repository ppa:ignaciovizzo/opencv3-nonfree -y'.format(
                sudo))
        os.system('{} apt update'.format(sudo))
        os.system(
            '{} DEBIAN_FRONTEND="noninteractive"  apt install libopencv-dev -y'
            .format(sudo))

        ocv = cmd_result('which opencv_version')
        if ocv is None or len(ocv) < 1:
            print('Check ocv failed.')
            return -1, envs
        print('success')

    # print all

    print('ubuntu \t\t:{}'.format(ubuntu))

    # check python
    print('python bin\t:{}'.format(cmd_result('which python3')))
    print('python version\t:{}'.format(
        cmd_result("python3 --version | awk '{print $2}'")))

    print('cmake bin\t:{}'.format(cmake))
    print('cmake version\t:{}'.format(
        cmd_result("cmake --version | head -n 1 | awk '{print $3}'")))

    print('make bin\t:{}'.format(make))
    print('make version\t:{}'.format(
        cmd_result(" make --version  | head -n 1 | awk '{print $3}' ")))

    print('wget bin\t:{}'.format(wget))
    print('g++-7 bin\t:{}'.format(gplus))

    print('mmcv version\t:{}'.format(mmcv_version))
    if mmcv_version is None:
        print('\t please install an mm serials algorithm later.')
        time.sleep(2)

    print('torch version\t:{}'.format(torch_version))
    if torch_version is None:
        print('\t please install pytorch later.')
        time.sleep(2)

    print('ocv version\t:{}'.format(cmd_result('opencv_version')))

    print('git bin\t\t:{}'.format(git))
    print('git version\t:{}'.format(
        cmd_result("git --version | awk '{print $3}' ")))
    print('protoc \t:{}'.format(cmd_result('{} --version'.format(protoc))))
    # work dir
    print('work dir \t:{}'.format(work_dir))
    # dep dir
    print('dep dir \t:{}'.format(dep_dir))

    print('\n')
    return 0, envs


def install_pyncnn(dep_dir):
    print('-' * 10 + 'build and install pyncnn' + '-' * 10)
    time.sleep(2)

    # generate unzip and build dir
    os.chdir(dep_dir)

    # git clone
    if not os.path.exists('ncnn'):
        os.system(
            'git clone https://github.com/tencent/ncnn && cd ncnn && git checkout 20220729'  # noqa: E501
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
    work_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
    dep_dir = os.path.abspath(os.path.join(work_dir, '..', 'mmdeploy-dep'))
    if not os.path.exists(dep_dir):
        if os.path.isfile(dep_dir):
            print('{} already exists and it is a file, exit.'.format(work_dir))
            return -1
        os.mkdir(dep_dir)

    success, envs = ensure_env(work_dir, dep_dir)
    if success != 0:
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
