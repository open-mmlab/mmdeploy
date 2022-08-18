# 一键式脚本安装

通过用户调研，我们得知多数使用者在了解 mmdeploy 前，已经熟悉 python 和 torch 用法。因此我们提供脚本简化 mmdeploy 安装。

假设您已经准备好 Python3.6 pip 以上环境（无论 conda 或 pyenv），运行这个脚本来安装 mmdeploy + ncnn backend

```bash
$ cd /path/to/mmdeploy
$ python3 tools/scripts/build_ubuntu_x64_ncnn.py
..
```

期间可能需要 sudo 密码，脚本会尽最大努力完成 mmdeploy SDK 和 demo：

- 检测系统版本、是否 root 用户，尝试修复 pip
- 寻找必须的基础工具，如 g++-7、cmake、wget 等
- 编译必须的依赖，如 pyncnn、 protobuf

脚本也会尽量避免影响 host 环境：

- 源码编译的依赖，都放在与 mmdeploy 同级的 `mmdeploy-dep` 目录中
- 不会主动修改 PATH、LD_LIBRARY_PATH、PYTHONPATH 等变量

这是已验证的安装脚本：

|          script          | OS version  |
| :----------------------: | :---------: |
| build_ubuntu_x64_ncnn.py | 18.04/20.04 |
