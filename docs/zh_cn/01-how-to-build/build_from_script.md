# 一键式脚本安装

通过用户调研，我们得知多数使用者在了解 mmdeploy 前，已经熟悉 python 和 torch 用法。因此我们提供脚本简化 mmdeploy 编译和安装。

假设您已经准备好 ubuntu Python3.6 pip 以上环境（无论 conda 或 pyenv），运行这个脚本来安装 mmdeploy + ncnn backend

```bash
$ cd /path/to/mmdeploy
$ python3 tools/scripts/build_ubuntu_x64_ncnn.py
..
```

脚本会尽最大努力完成 mmdeploy 编译安装：

- 检测 ubuntu 版本、是否 root 用户，尝试修复 pip 错误
- 寻找必须的基础工具，如 g++-7、cmake、wget 等
- 源码编译必须的依赖，如 pyncnn、 protobuf

脚本也会尽量避免修改 host 环境：

- 源码编译的依赖，都放在与 mmdeploy 同级的 `mmdeploy-dep` 目录中
- 不会主动修改 PATH、LD_LIBRARY_PATH、PYTHONPATH 等变量

这些是已验证的安装脚本：

|          script          |    OS version     |
| :----------------------: | :---------------: |
| build_ubuntu_x64_ncnn.py | 16.04/18.04/20.04 |
