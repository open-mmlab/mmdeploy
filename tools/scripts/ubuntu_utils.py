# Copyright (c) OpenMMLab. All rights reserved.
import os


def cmd_result(txt: str):
    cmd = os.popen(txt)
    return cmd.read().rstrip().lstrip()


def version_major(txt: str) -> int:
    return int(txt.split('.')[0])


def version_minor(txt: str) -> int:
    return int(txt.split('.')[1])


def cu_version_name(version: str) -> str:
    versions = version.split('.')
    return 'cu' + versions[0] + versions[1]


def simple_check_install(bin: str, sudo: str) -> str:
    result = cmd_result('which {}'.format(bin))
    if result is None or len(result) < 1:
        print('{} not found, try install {} ..'.format(bin, bin), end='')
        os.system('{} apt install {} -y'.format(sudo, bin))
        result = cmd_result('which {}'.format(bin))
        if result is None or len(result) < 1:
            print('Check {} failed.'.format(bin))
            return None
        print('success')
    return result
