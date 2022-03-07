## 安装 MMdeploy


### 下载代码仓库 MMDeploy

  ```bash
  git clone -b master git@github.com:open-mmlab/mmdeploy.git MMDeploy
  cd MMDeploy
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

- 安装 mmcv-full, 更多安装方式可查看[教程](https://github.com/open-mmlab/mmcv#installation)

    ```bash
    export cu_version=cu111 # cuda 11.1
    export torch_version=torch1.8
    pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/${cu_version}/${torch_version}/index.html
    ```

### 安装推理引擎

您可以根据自身需求，构建和安装如下推理引擎：

- [ONNX Runtime](https://mmdeploy.readthedocs.io/en/latest/backends/onnxruntime.html)
- [TensorRT](https://mmdeploy.readthedocs.io/en/latest/backends/tensorrt.html)
- [ncnn](https://mmdeploy.readthedocs.io/en/latest/backends/ncnn.html)
- [PPLNN](https://mmdeploy.readthedocs.io/en/latest/backends/pplnn.html)
- [OpenVINO](https://mmdeploy.readthedocs.io/en/latest/backends/openvino.html)
- [TorchScript](https://mmdeploy.readthedocs.io/en/latest/backends/torchscript.md)

### 安装 MMDeploy

```bash
cd ${MMDEPLOY_DIR} # 切换至项目根目录
pip install -e .
```

**Note**

- 有些依赖项是可选的。运行 `pip install -e .` 将进行最小化依赖安装。 如果需安装其他可选依赖项，请执行`pip install -r requirements/optional.txt`，
或者 `pip install -e . [optional]`。其中，`[optional]`可以填写`all`, `tests`, `build`, `optional`

### 构建 SDK

读者如果只对模型转换感兴趣，那么可以跳过本章节

#### 安装依赖项

目前，SDK在Linux-x86_64经过测试验证，未来将加入对更多平台的支持。 使用SDK，需要安装若干依赖包。本文以 Ubuntu 18.04为例，逐一介绍各依赖项的安装方法

- OpenCV 3+

  ```bash
  git clone -b master https://github.com/open-mmlab/mmdeploy.git MMDeploy
  cd MMDeploy
  git submodule update --init --recursive
  ```
### 编译 MMDeploy
根据您的目标平台，点击如下对应的链接，按照说明编译 MMDeploy
- [Linux-x86_64](build/linux.md)
- [Windows](build/windows.md)
- [Android-aarch64](build/android.md)
- [NVIDIA Jetson](../en/tutorials/how_to_install_mmdeploy_on_jetsons.md)
