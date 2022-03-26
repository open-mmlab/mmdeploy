# Build MMDeploy

## Download MMDeploy


  ```bash
  git clone -b master git@github.com:open-mmlab/mmdeploy.git MMDeploy
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
  - If it fails when `git clone` via `SSH`, you can try the `HTTPS` protocol like this:

    ```bash
    git clone -b master https://github.com/open-mmlab/mmdeploy.git MMDeploy
    cd MMDeploy
    git submodule update --init --recursive
    ```

## Build MMDeploy
Please visit the following links to find out how to build MMDeploy according to the target platform.
- [Linux-x86_64](build/linux.md)
- [Windows](build/windows.md)
- [Android-aarch64](build/android.md)
- [NVIDIA Jetson](tutorials/how_to_install_mmdeploy_on_jetsons.md)
