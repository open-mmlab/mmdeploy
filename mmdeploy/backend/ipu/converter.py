# Copyright (c) OpenMMLab. All rights reserved.
import subprocess


def onnx_to_popef(onnx_path, ipu_config):
    command = [
        'python3', '-m', 'poprt.cli', '--input_model', onnx_path,
        '--export_popef', '--convert_version', '11'
    ]
    for key in ipu_config.keys():
        if key == 'type':
            continue
        if key == 'popart_options':
            opt_dict = ipu_config[key]
            command += ['--' + str(key)]
            for ikey in opt_dict.keys():
                command += [str(ikey) + '=' + str(opt_dict[ikey])]

        elif ipu_config[key] == '':
            command += ['--' + str(key)]
        else:
            command += ['--' + str(key), str(ipu_config[key])]

    # print('command ', command)
    if subprocess.call(command) != 0:
        raise RuntimeError(
            '\n\n!!! PopConverter compile command failed, please check the above trace for details.'
        )
