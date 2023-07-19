# 源码手动安装

如果网络良好，我们建议使用 [docker](build_from_docker.md) 或 [一键式脚本](build_from_script.md) 方式。

## 下载

```bash
git clone -b main git@github.com:open-mmlab/mmdeploy.git --recursive
```

### FAQ

- 如果由于网络等原因导致拉取仓库子模块失败，可以尝试通过如下指令手动再次安装子模块:

  ```bash
  git clone git@github.com:NVIDIA/cub.git third_party/cub
  cd third_party/cub
  git checkout c3cceac115

  # 返回至 third_party 目录, 克隆 pybind11
  cd ..
  git clone git@github.com:pybind/pybind11.git pybind11
  cd pybind11
  git checkout 70a58c5

  cd ..
  git clone git@github.com:gabime/spdlog.git spdlog
  cd spdlog
  git checkout 9e8e52c048
  ```

- 如果以 `SSH` 方式 `git clone` 代码失败，您可以尝试使用 `HTTPS` 协议下载代码：

  ```bash
  git clone -b main https://github.com/open-mmlab/mmdeploy.git MMDeploy
  cd MMDeploy
  git submodule update --init --recursive
  ```

## 编译

根据您的目标平台，点击如下对应的链接，按照说明编译 MMDeploy

- [Linux-x86_64](linux-x86_64.md)
- [Windows](windows.md)
- [MacOS](macos-arm64.md)
- [Android-aarch64](android.md)
- [NVIDIA Jetson](jetsons.md)
- [Qcom SNPE](snpe.md)
- [RISC-V](riscv.md)
- [Rockchip](rockchip.md)
