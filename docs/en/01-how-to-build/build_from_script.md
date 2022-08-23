# Build from Script

Through user investigation, we know that most users are already familiar with python and torch before using mmdeploy. Therefore we provide scripts to simplify mmdeploy installation.

Assuming you have a python ready (whether `conda` or `pyenv`), run this script to install mmdeploy + ncnn backend, `nproc` is not compulsory.

```bash
$ cd /path/to/mmdeploy
$ python3 tools/scripts/build_ubuntu_x64_ncnn.py $(nproc)
..
```

A sudo password may be required during this time, and the script will try its best to build and install mmdeploy SDK and demo:

- Detect host OS version, `make` job number, whether use `root` and try to fix `python3 -m pip`
- Find the necessary basic tools, such as g++-7, cmake, wget, etc.
- Compile necessary dependencies, such as pyncnn, protobuf

The script will also try to avoid affecting host environment:

- The dependencies of source code compilation are placed in the `mmdeploy-dep` directory at the same level as mmdeploy
- The script would not modify variables such as PATH, LD_LIBRARY_PATH, PYTHONPATH, etc.

Here is the verified installation script. If you want mmdeploy to support multiple backends at the same time, you can execute each script once:

|             script              | OS version  |
| :-----------------------------: | :---------: |
|    build_ubuntu_x64_ncnn.py     | 18.04/20.04 |
|     build_ubuntu_x64_ort.py     | 18.04/20.04 |
|    build_ubuntu_x64_pplnn.py    | 18.04/20.04 |
| build_ubuntu_x64_torchscript.py | 18.04/20.04 |
