## 安装 MMdeploy


### 下载代码仓库 MMDeploy

  ```bash
  git clone -b master git@github.com:open-mmlab/mmdeploy.git MMDeploy
  cd MMDeploy
  export MMDEPLOY_DIR=$(pwd)
  git submodule update --init --recursive
  ```

  提示:

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
    ```

### 编译 MMDeploy
根据您的目标平台，点击如下对应的链接，按照说明编译 MMDeploy
- [Linux-x86_64](build/linux.md)
- [Windows](build/windows.md)
