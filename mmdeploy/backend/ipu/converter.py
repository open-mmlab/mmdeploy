# Copyright (c) OpenMMLab. All rights reserved.
import subprocess
import json
import os


def onnx_to_popef(onnx_path, ipu_config):
    command = [
        'python3', '-m', 'poprt.cli', '--input_model', onnx_path,
        '--export_popef', '--convert_version', '11'
    ]
    bps = 1
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

        if key == "batches_per_step":
            bps = int(ipu_config[key])

    # print('command ', command)
    if subprocess.call(command) != 0:
        raise RuntimeError('\n\n!!! PopConverter compile command failed, \
                please check the above trace for details.')

    onnx_path = os.path.abspath(onnx_path)
    config_dir = onnx_path[:onnx_path.rfind('/')]
    ipu_json_path = os.path.join(config_dir, 'ipu_params.json')
    ipu_param_json = {"batches_per_step": bps}
    with open(ipu_json_path, 'w') as f:
        json.dump(ipu_param_json, f)
    print("dumped ipu param json ", ipu_json_path)
