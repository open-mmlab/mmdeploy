## Frequently Asked Questions

### TensorRT

- "WARNING: Half2 support requested on hardware without native FP16 support, performance will be negatively affected."

  Fp16 mode requires a device with full-rate fp16 support.

- "error: parameter check failed at: engine.cpp::setBindingDimensions::1046, condition: profileMinDims.d\[i\] \<= dimensions.d\[i\]"

  When building an `ICudaEngine` from an `INetworkDefinition` that has dynamically resizable inputs, users need to specify at least one optimization profile. Which can be set in deploy config:

  ```python
  backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 320, 320],
                    opt_shape=[1, 3, 800, 1344],
                    max_shape=[1, 3, 1344, 1344])))
    ])
  ```

  The input tensor shape should be limited between `min_shape` and `max_shape`.

- "error: \[TensorRT\] INTERNAL ERROR: Assertion failed: cublasStatus == CUBLAS_STATUS_SUCCESS"

  TRT 7.2.1 switches to use cuBLASLt (previously it was cuBLAS). cuBLASLt is the defaulted choice for SM version >= 7.0. You may need CUDA-10.2 Patch 1 (Released Aug 26, 2020) to resolve some cuBLASLt issues. Another option is to use the new TacticSource API and disable cuBLASLt tactics if you dont want to upgrade.

### Libtorch

- Error: `libtorch/share/cmake/Caffe2/Caffe2Config.cmake:96 (message):Your installed Caffe2 version uses cuDNN but I cannot find the cuDNN libraries.  Please set the proper cuDNN prefixes and / or install cuDNN.`

  May `export CUDNN_ROOT=/root/path/to/cudnn` to resolve the build error.

### Windows

- Error: similar like this `OSError: [WinError 1455] The paging file is too small for this operation to complete. Error loading "C:\Users\cx\miniconda3\lib\site-packages\torch\lib\cudnn_cnn_infer64_8.dll" or one of its dependencies`

  Solution: according to this [post](https://stackoverflow.com/questions/64837376/how-to-efficiently-run-multiple-pytorch-processes-models-at-once-traceback), the issue may be caused by NVidia and will fix in *CUDA release 11.7*. For now one could use the [fixNvPe.py](https://gist.github.com/cobryan05/7d1fe28dd370e110a372c4d268dcb2e5) script to modify the nvidia dlls in the pytorch lib dir.

  `python fixNvPe.py --input=C:\Users\user\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\lib\*.dll`

  You can find your pytorch installation path with:

  ```python
  import torch
  print(torch.__file__)
  ```

- enable_language(CUDA) error

  ```
  -- Selecting Windows SDK version 10.0.19041.0 to target Windows 10.0.19044.
  -- Found CUDA: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1 (found version "11.1")
  CMake Error at C:/Software/cmake/cmake-3.23.1-windows-x86_64/share/cmake-3.23/Modules/CMakeDetermineCompilerId.cmake:491 (message):
    No CUDA toolset found.
  Call Stack (most recent call first):
    C:/Software/cmake/cmake-3.23.1-windows-x86_64/share/cmake-3.23/Modules/CMakeDetermineCompilerId.cmake:6 (CMAKE_DETERMINE_COMPILER_ID_BUILD)
    C:/Software/cmake/cmake-3.23.1-windows-x86_64/share/cmake-3.23/Modules/CMakeDetermineCompilerId.cmake:59 (__determine_compiler_id_test)
    C:/Software/cmake/cmake-3.23.1-windows-x86_64/share/cmake-3.23/Modules/CMakeDetermineCUDACompiler.cmake:339 (CMAKE_DETERMINE_COMPILER_ID)
    C:/workspace/mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0/sdk/lib/cmake/MMDeploy/MMDeployConfig.cmake:27 (enable_language)
    CMakeLists.txt:5 (find_package)
  ```

  **Cause：** CUDA Toolkit 11.1 was installed before Visual Studio, so the VS plugin was not installed. Or the version of VS is too new, so that the installation of the VS plugin is skipped during the installation of the CUDA Toolkit

  **Solution：** This problem can be solved by manually copying the four files in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\extras\visual_studio_integration\MSBuildExtensions` to `C:\Software\Microsoft Visual Studio\2022\Community\Msbuild\Microsoft\VC\v170\BuildCustomizations` The specific path should be changed according to the actual situation.

### ONNX Runtime

- Under Windows system, when visualizing model inference result failed with the following error:
  ```
  onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Failed to load library, error code: 193
  ```
  **Cause：** In latest Windows systems, there are two `onnxruntime.dll` under the system path, and they will be loaded first, causing conflicts.
  ```
  C:\Windows\SysWOW64\onnxruntime.dll
  C:\Windows\System32\onnxruntime.dll
  ```
  **Solution：** Choose one of the following two options
  1. Copy the dll in the lib directory of the downloaded onnxruntime to the directory where mmdeploy_onnxruntime_ops.dll locates (It is recommended to use Everything to search the ops dll). For example, copy [`onnxruntime`](https://github.com/microsoft/onnxruntime/releases/tag/v1.8.1) `lib/onnxruntime.dll` to `mmdeploy/lib`, then the `mmdeploy/lib` directory should like this
     ```
     `-- mmdeploy_onnxruntime_ops.dll
     `-- mmdeploy_onnxruntime_ops.lib
     `-- onnxruntime.dll
     ```
  2. Rename the two dlls in the system path so that they cannot be loaded.

### Pip

- pip installed package but could not `import` them.

  Make sure your are using conda pip.

  ```bash
  $ which pip
  # /path/to/.local/bin/pip
  /path/to/miniconda3/lib/python3.9/site-packages/pip
  ```
