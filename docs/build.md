## Build MMDeploy

### Preparation

- Download MMDeploy

    ```bash
    git clone -b master git@github.com:grimoire/deploy_prototype.git MMDeploy
    cd MMDeploy
    git submodule update --init --recursive
    ```

    Note:

  - If fetching submodule fails, you could get submodule manually by following instructions:

      ```bash
      git clone git@github.com:NVIDIA/cub.git third_party/cub
      cd third_party/cub
      git checkout c3cceac115

      # go back to third_party directory and git clone pybind11
      cd ..
      git clone git@github.com:pybind/pybind11.git pybind11
      cd pybind11
      git checkout 70a58c5
      ```

- Install cmake

    Install cmake>=3.14.0, you could refer to [cmake website](https://cmake.org/install) for more detailed info.

    ```bash
    apt-get install -y libssl-dev
    wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
    tar -zxvf cmake-3.20.0.tar.gz
    cd cmake-3.20.0
    ./bootstrap
    make
    make install
    ```

### Build backend support

Build the inference engine extension libraries you need.

- [ONNX Runtime](backends/onnxruntime.md)
- [TensorRT](backends/tensorrt.md)
- [ncnn](backends/ncnn.md)
- [PPL](backends/ppl.md)
- [OpenVINO](backends/openvino.md)

### Install mmdeploy

```bash
cd ${MMDEPLOY_DIR} # To mmdeploy root directory
pip install -e .
```
Some dependencies are optional. Simply running `pip install -e .` will only install the minimum runtime requirements.
To use optional dependencies install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -e .[optional]`).
Valid keys for the extras field are: `all`, `tests`, `build`, `optional`.
