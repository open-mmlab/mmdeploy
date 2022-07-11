# FAQ

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

### Pip

- pip installed package but could not `import` them.

  Make sure your are using conda pip.

  ```bash
  $ which pip
  # /path/to/.local/bin/pip
  /path/to/miniconda3/lib/python3.9/site-packages/pip
  ```
