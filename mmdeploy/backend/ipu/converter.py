import popconverter
import sys


def onnx_to_popef(onnx_path, popef_path, ipu_config):
    command = ['python3', '-m', 'popconverter.cli',
               '--input_model', onnx_path,
               #  '--output_dir', output_dir,
               '--output_popef', popef_path,
               '--convert_version', '11']
    for key in ipu_config.keys():
        command += ['--'+str(key)+' '+str(ipu_config[key])]

    sys.cmd(command)
