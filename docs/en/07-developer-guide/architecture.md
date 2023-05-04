# mmdeploy Architecture

This article mainly introduces the functions of each directory of mmdeploy and how it works from model conversion to real inference.

## Take a general look at the directory structure

The entire mmdeploy can be seen as two independent parts: model conversion and SDK.

We introduce the entire repo directory structure and functions, without having to study the source code, just have an impression.

Peripheral directory features:

```bash
$ cd /path/to/mmdeploy
$ tree -L 1
.
├── CMakeLists.txt    # Compile custom operator and cmake configuration of SDK
├── configs                   # Algorithm library configuration for model conversion
├── csrc                          # SDK and custom operator
├── demo                      # FFI interface examples in various languages, such as csharp, java, python, etc.
├── docker                   # docker build
├── mmdeploy           # python package for model conversion
├── requirements      # python requirements
├── service                    # Some small boards not support python, we use C/S mode for model conversion, here is server code
├── tests                         # unittest
├── third_party           # 3rd party dependencies required by SDK and FFI
└── tools                        # Tools are also the entrance to all functions, such as onnx2xx.py, profiler.py, test.py, etc.
```

It should be clear

- Model conversion mainly depends on `tools`, `mmdeploy` and small part of `csrc` directory;
- SDK is consist of three directories: `csrc`, `third_party` and `demo`.

## Model Conversion

Here we take ViT of mmpretrain as model example, and take ncnn as inference backend example. Other models and inferences are similar.

Let's take a look at the mmdeploy/mmdeploy directory structure and get an impression:

```bash
.
├── apis                             # The api used by tools is implemented here, such as onnx2ncnn.py
│   ├── calibration.py          # trt dedicated collection of quantitative data
│   ├── core                              # Software infrastructure
│   ├── extract_model.py  # Use it to export part of onnx
│   ├── inference.py             # Abstract function, which will actually call torch/ncnn specific inference
│   ├── ncnn                            # ncnn Wrapper
│   └── visualize.py              # Still an abstract function, which will actually call torch/ncnn specific inference and visualize
..
├── backend                  # Backend wrapper
│   ├── base                            # Because there are multiple backends, there must be an OO design for the base class
│   ├── ncnn                           # This calls the ncnn python interface for model conversion
│   │   ├── init_plugins.py           # Find the path of ncnn custom operators and ncnn tools
│   │   ├── onnx2ncnn.py            # Wrap `mmdeploy_onnx2ncnn` into a python interface
│   │   ├── quant.py                       # Wrap `ncnn2int8` as a python interface
│   │   └── wrapper.py                  # Wrap pyncnn forward API
..
├── codebase                #  Algorithm rewriter
│   ├── base                          # There are multiple algorithms here that we need a bit of OO design
│   ├── mmpretrain                      #  mmpretrain related model rewrite
│   │   ├── deploy                       # mmpretrain implementation of base abstract task/model/codebase
│   │   └── models                      # Real model rewrite
│   │       ├── backbones                 # Rewrites of backbone network parts, such as multiheadattention
│   │       ├── heads                           # Such as MultiLabelClsHead
│   │       ├── necks                            # Such as GlobalAveragePooling
│..
├── core                         # Software infrastructure of rewrite mechanism
├── mmcv                     #  Rewrite mmcv
├── pytorch                 #  Rewrite pytorch operator for ncnn, such as Gemm
..
```

Each line above needs to be read, don't skip it.

When typing `tools/deploy.py` to convert ViT, these are 3 things:

1. Rewrite of mmpretrain ViT forward
2. ncnn does not support `gather` opr, customize and load it with libncnn.so
3. Run exported ncnn model with real inference, render output, and make sure the result is correct

### 1. Rewrite `forward`

Because when exporting ViT to onnx, it generates some operators that ncnn doesn't support perfectly, mmdeploy's solution is to hijack the forward code and change it. The output onnx is suitable for ncnn.

For example, rewrite the process of `conv -> shape -> concat_const -> reshape` to `conv -> reshape` to trim off the redundant `shape` and `concat` operator.

All mmpretrain algorithm rewriters are in the `mmdeploy/codebase/mmpretrain/models` directory.

### 2. Custom Operator

Operators customized for ncnn are in the `csrc/mmdeploy/backend_ops/ncnn/` directory, and are loaded together with `libncnn.so` after compilation. The essence is in hotfix ncnn, which currently implements these operators:

- topk
- tensorslice
- shape
- gather
- expand
- constantofshape

### 3. Model Conversion and testing

We first use the modified `mmdeploy_onnx2ncnn`to convert model, then inference with`pyncnn` and custom ops.

When encountering a framework such as snpe that does not support python well, we use C/S mode: wrap a server with protocols such as gRPC, and forward the real inference output.

For Rendering, mmdeploy directly uses the rendering API of upstream algorithm codebase.

## SDK

After the model conversion completed, the SDK compiled with C++ can be used to execute on different platforms.

Let's take a look at the csrc/mmdeploy directory structure:

```bash
.
├── apis           # csharp, java, go, Rust and other FFI interfaces
├── backend_ops    # Custom operators for each inference framework
├── CMakeLists.txt
├── codebase       # The type of results preferred by each algorithm framework, such as multi-use bbox for detection task
├── core           # Abstraction of graph, operator, device and so on
├── device         # Implementation of CPU/GPU device abstraction
├── execution      # Implementation of the execution abstraction
├── graph          # Implementation of graph abstraction
├── model          # Implement both zip-compressed and uncompressed work directory
├── net            # Implementation of net, such as wrap ncnn forward C API
├── preprocess     # Implement preprocess
└── utils          # OCV tools
```

The essence of the SDK is to design a set of abstraction of the computational graph, and combine the **multiple models'**

- preprocess
- inference
- postprocess

Provide FFI in multiple languages at the same time.
