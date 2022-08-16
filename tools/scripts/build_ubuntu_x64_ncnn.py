# Copyright (c) OpenMMLab. All rights reserved.
import os


def cmd_result(txt: str):
    cmd = os.popen(txt)
    return cmd.read().rstrip().lstrip()


def version_major(txt: str) -> int:
    return int(txt.split('.')[0])


def check_env(work_dir) -> int:
    """check python, cmake and torch environment.

    Returns:
        int: _description_
    """
    envs = []
    print('-' * 10 + 'check env' + '-' * 10)

    # check ubuntu
    ubuntu = cmd_result(
        """ lsb_release -a 2>/dev/null | grep "Release" | tail -n 1 | awk '{print $NF}' """  # noqa: E501
    )
    print('ubuntu \t:{}'.format(ubuntu))

    # check python
    print('python bin\t:{}'.format(cmd_result('which python3')))
    print('python version\t:{}'.format(
        cmd_result("python3 --version | awk '{print $2}'")))

    # check cmake version
    cmake = cmd_result('which cmake')
    if cmake is None or len(cmake) < 1:
        os.system('python3 -m pip install cmake >=3.14.0')

    cmake = cmd_result('which cmake')
    if cmake is None or len(cmake) < 1:
        env = 'export PATH=${PATH}:~/.local/bin'
        os.system(env)
        envs.append(env)

    cmake = cmd_result('which cmake')
    if cmake is None or len(cmake) < 1:
        print('Check cmake failed.')
        return -1

    print('cmake bin\t:{}'.format(cmake))
    print('cmake version\t:{}'.format(
        cmd_result("cmake --version | head -n 1 | awk '{print $3}'")))

    # check g++ version
    gplus = cmd_result('which g++-7')
    if gplus is None or len(gplus) < 1:
        # install g++
        if version_major(ubuntu) <= 18:
            os.system('sudo add-apt-repository ppa:ubuntu-toolchain-r/test')
            os.system('sudo apt update --fix-missing')
        os.system('sudo apt install gcc-7 g++-7')

    gplus = cmd_result('which g++-7')
    if gplus is None or len(gplus) < 1:
        print('Check g++-7 failed.')
        return -1

    print('g++-7 bin\t:{}'.format(gplus))

    # check torch and mmcv, it is not compulsory
    mmcv_version = None
    try:
        import mmcv
        mmcv_version = mmcv.__version__
    except Exception:
        pass
    print('mmcv version\t:{}'.format(mmcv_version))

    torch_version = None
    try:
        import torch
        torch_version = torch.__version__
    except Exception:
        pass
    print('torch version\t:{}'.format(torch_version))

    # wget
    wget = cmd_result('which wget')
    if wget is None or len(wget) < 1:
        os.system('sudo apt install wget')
    wget = cmd_result('which wget')
    if wget is None or len(wget) < 1:
        print('Check wget failed.')
        return -1
    print('wget bin\t:{}'.format(wget))

    # work dir
    print('work dir \t:{}'.format(work_dir))
    print('\n')
    return 0


def install_pyncnn(work_dir):
    print('-' * 10 + 'build and install pyncnn' + '-' * 10)

    return 0


def main():
    work_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..', '..'))

    if check_env(work_dir) != 0:
        return -1

    if install_pyncnn(work_dir) != 0:
        return -1


if __name__ == '__main__':
    main()
