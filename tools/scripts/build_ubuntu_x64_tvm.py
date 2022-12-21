# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import sys
import time

from ubuntu_utils import cmd_result, ensure_base_env, get_job


def install_llvm(dep_dir):
    print('-' * 10 + 'install llvm' + '-' * 10)

    os.chdir(dep_dir)
    os.system(
        'wget --no-check-certificate -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -'  # noqa: E501
    )

    ubuntu = cmd_result(
        """ lsb_release -a 2>/dev/null | grep "Release" | tail -n 1 | awk '{print $NF}' """  # noqa: E501
    )

    nickname_dict = {
        '18.04': 'bionic',
        '20.04': 'focal',
        '22.04': 'jammy',
        '22.10': 'kinetic'
    }
    nickname = nickname_dict.get(ubuntu, None)
    if nickname is None:
        raise NotImplementedError(f'Unsupported ubuntu version {ubuntu}.')
    os.system(
        f"add-apt-repository 'deb http://apt.llvm.org/{nickname}/   llvm-toolchain-{nickname}-10  main'"  # noqa: E501
    )
    os.system('sudo apt update')
    os.system(
        'sudo apt-get install llvm-10 lldb-10 llvm-10-dev libllvm10 llvm-10-runtime'  # noqa: E501
    )


def install_tvm(dep_dir):
    print('-' * 10 + 'build and install tvm' + '-' * 10)
    time.sleep(2)

    os.system('sudo apt-get update')
    os.system(
        'sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev'  # noqa: E501
    )

    # generate unzip and build dir
    os.chdir(dep_dir)

    # git clone
    if not osp.exists('tvm'):
        os.system(
            'git clone --branch v0.10.0 --depth 1 --recursive https://github.com/apache/tvm tvm'  # noqa: E501
        )

    tvm_dir = osp.join(dep_dir, 'tvm')
    os.chdir(tvm_dir)

    # build
    if not osp.exists('build'):
        os.system('mkdir build')
    os.system('cp cmake/config.cmake build')

    os.chdir(osp.join(tvm_dir, 'build'))

    os.system(
        """ sed -i "s@set(USE_LLVM OFF)@set(USE_LLVM /usr/bin/llvm-config-10)@g" config.cmake """  # noqa: E501
    )

    os.system('cmake .. && make -j {} && make runtime'.format(g_jobs))

    # set env
    os.system(
        """ echo 'export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH' >> ~/mmdeploy.env """  # noqa: E501
        .format(os.path.join(tvm_dir, 'build')))

    # install python package
    os.chdir(osp.join(tvm_dir, 'python'))
    os.system(""" python3 setup.py install --user """)

    # install dependency
    os.system(
        """ python3 -m pip install xgboost decorator psutil scipy attrs tornado """  # noqa: E501
    )

    return tvm_dir


def install_mmdeploy(work_dir, tvm_dir):
    print('-' * 10 + 'build and install mmdeploy' + '-' * 10)
    time.sleep(3)

    os.chdir(work_dir)
    os.system('git submodule init')
    os.system('git submodule update')

    if not os.path.exists('build'):
        os.system('mkdir build')

    os.system('rm -rf build/CMakeCache.txt')

    cmd = 'cd build && cmake ..'
    cmd += ' -DMMDEPLOY_BUILD_SDK=ON '
    cmd += ' -DMMDEPLOY_BUILD_EXAMPLES=ON '
    cmd += ' -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON '
    cmd += ' -DMMDEPLOY_TARGET_DEVICES=cpu '
    cmd += ' -DMMDEPLOY_TARGET_BACKENDS=tvm '
    cmd += ' -DTVM_DIR={} '.format(tvm_dir)
    os.system(cmd)

    os.system('cd build && make -j {} && make install'.format(g_jobs))
    os.system('python3 -m pip install -v -e .')
    os.system(""" echo 'export PATH={}:$PATH' >> ~/mmdeploy.env """.format(
        os.path.join(work_dir, 'mmdeploy', 'backend', 'tvm')))
    try:
        import mmcv
        print(mmcv.__version__)
        os.system('python3 tools/check_env.py')
    except Exception:
        print('Please install torch & mmcv later...')
    return 0


def main():
    """Auto install mmdeploy with tvm. To verify this script:

    1) use `sudo docker run -v /path/to/mmdeploy:/root/mmdeploy -v /path/to/Miniconda3-latest-Linux-x86_64.sh:/root/miniconda.sh -it ubuntu:18.04 /bin/bash` # noqa: E501
    2) install conda and setup python environment
    3) run `python3 tools/scripts/build_ubuntu_x64_tvm.py`

    Returns:
        _type_: _description_
    """
    global g_jobs
    g_jobs = get_job(sys.argv)
    print('g_jobs {}'.format(g_jobs))

    work_dir = osp.abspath(osp.join(__file__, '..', '..', '..'))
    dep_dir = osp.abspath(osp.join(work_dir, '..', 'mmdeploy-dep'))
    if not osp.exists(dep_dir):
        if osp.isfile(dep_dir):
            print('{} already exists and it is a file, exit.'.format(work_dir))
            return -1
        os.mkdir(dep_dir)

    success = ensure_base_env(work_dir, dep_dir)
    if success != 0:
        return -1

    install_llvm(dep_dir)
    tvm_dir = install_tvm(dep_dir)
    if install_mmdeploy(work_dir, tvm_dir) != 0:
        return -1

    if osp.exists('~/mmdeploy.env'):
        print('Please source ~/mmdeploy.env to setup your env !')
        os.system('cat ~/mmdeploy.env')


if __name__ == '__main__':
    main()
