# 一键式脚本安装

通过用户调研，我们得知多数使用者在了解 mmdeploy 前，已经熟知 python 和 torch 用法。因此我们提供脚本简化 mmdeploy 安装。

假设您已经准备好

- python3 -m pip（必须，conda 或 pyenv 皆可）
- nvcc（取决于推理后端）
- torch（非必须，可延后安装）

运行这个脚本来安装 mmdeploy + ncnn backend，`nproc` 可以不指定。

```bash
$ cd /path/to/mmdeploy
$ python3 tools/scripts/build_ubuntu_x64_ncnn.py
..
```

期间可能需要 sudo 密码，脚本会尽最大努力完成 mmdeploy SDK 和 demo：

- 检测系统版本、make 使用的 job 个数、是否 root 用户，也会自动修复 pip 问题
- 寻找必须的基础工具，如 g++-7、cmake、wget 等
- 编译必须的依赖，如 pyncnn、 protobuf

脚本也会尽量避免影响 host 环境：

- 源码编译的依赖，都放在与 mmdeploy 同级的 `mmdeploy-dep` 目录中
- 不会主动修改 PATH、LD_LIBRARY_PATH、PYTHONPATH 等变量
- 会打印需要修改的环境变量，**需要注意最终的输出信息**

脚本最终会执行 `python3 tools/check_env.py`，安装成功应显示对应 backend 的版本号和 `ops_is_available: True`，例如：

```bash
$ python3 tools/check_env.py
..
2022-09-13 14:49:13,767 - mmdeploy - INFO - **********Backend information**********
2022-09-13 14:49:14,116 - mmdeploy - INFO - onnxruntime: 1.8.0	ops_is_avaliable : True
2022-09-13 14:49:14,131 - mmdeploy - INFO - tensorrt: 8.4.1.5	ops_is_avaliable : True
2022-09-13 14:49:14,139 - mmdeploy - INFO - ncnn: 1.0.20220901	ops_is_avaliable : True
2022-09-13 14:49:14,150 - mmdeploy - INFO - pplnn_is_avaliable: True
..
```

这是已验证的安装脚本。如果想让 mmdeploy 同时支持多种 backend，每个脚本执行一次即可：

|             script              |     OS version      |
| :-----------------------------: | :-----------------: |
|    build_ubuntu_x64_ncnn.py     |     18.04/20.04     |
|     build_ubuntu_x64_ort.py     |     18.04/20.04     |
|    build_ubuntu_x64_pplnn.py    |     18.04/20.04     |
| build_ubuntu_x64_torchscript.py |     18.04/20.04     |
|  build_jetson_orin_python38.sh  | JetPack5.0 L4T 34.1 |
