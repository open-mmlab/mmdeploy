# Build from Source

## Download

```shell
git clone -b master git@github.com:open-mmlab/mmdeploy.git --recursive
```

Note:

- If fetching submodule fails, you could get submodule manually by following instructions:

  ```shell
  cd mmdeploy
  git clone git@github.com:NVIDIA/cub.git third_party/cub
  cd third_party/cub
  git checkout c3cceac115

  # go back to third_party directory and git clone pybind11
  cd ..
  git clone git@github.com:pybind/pybind11.git pybind11
  cd pybind11
  git checkout 70a58c5
  ```

- If it fails when `git clone` via `SSH`, you can try the `HTTPS` protocol like this:

  ```shell
  git clone -b master https://github.com/open-mmlab/mmdeploy.git --recursive
  ```

## Build

Please visit the following links to find out how to build MMDeploy according to the target platform.

- [Linux-x86_64](linux-x86_64.md)
- [Windows](windows.md)
- [Android-aarch64](android.md)
- [NVIDIA Jetson](jetsons.md)
- [SNPE](snpe.md)
- [RISC-V](riscv.md)
