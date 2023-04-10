# 第六章： TensorRT 模型构建与推理

模型部署入门教程继续更新啦！相信经过前几期的学习，大家已经对 ONNX 这一中间表示有了一个比较全面的认识，但是在具体的生产环境中，ONNX 模型常常需要被转换成能被具体推理后端使用的模型格式。本篇教程我们就和大家一起来认识大名鼎鼎的推理后端 TensorRT。

## TensorRT 简介

TensorRT 是由 NVIDIA 发布的深度学习框架，用于在其硬件上运行深度学习推理。TensorRT 提供量化感知训练和离线量化功能，用户可以选择 INT8 和 FP16 两种优化模式，将深度学习模型应用到不同任务的生产部署，如视频流、语音识别、推荐、欺诈检测、文本生成和自然语言处理。TensorRT 经过高度优化，可在 NVIDIA GPU 上运行， 并且可能是目前在 NVIDIA GPU 运行模型最快的推理引擎。关于 TensorRT 更具体的信息可以访问 [TensorRT官网](https://developer.nvidia.com/tensorrt) 了解。

## 安装 TensorRT

### Windows

默认在一台有 NVIDIA 显卡的机器上，提前安装好 [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) 和 [CUDNN](https://developer.nvidia.com/rdp/cudnn-archive)，登录 NVIDIA 官方网站下载和主机 CUDA 版本适配的 TensorRT 压缩包即可。

以 CUDA 版本是 10.2 为例，选择适配 CUDA 10.2 的 [zip 包](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.5.1/zip/tensorrt-8.2.5.1.windows10.x86_64.cuda-10.2.cudnn8.2.zip)，下载完成后，有 conda 虚拟环境的用户可以优先切换到虚拟环境中，然后在 powershell 中执行类似如下的命令安装并测试：

```shell
cd \the\path\of\tensorrt\zip\file
Expand-Archive TensorRT-8.2.5.1.Windows10.x86_64.cuda-10.2.cudnn8.2.zip .
$env:TENSORRT_DIR = "$pwd\TensorRT-8.2.5.1"
$env:path = "$env:TENSORRT_DIR\lib;" + $env:path
pip install $env:TENSORRT_DIR\python\tensorrt-8.2.5.1-cp36-none-win_amd64.whl
python -c "import tensorrt;print(tensorrt.__version__)"
```

上述命令会在安装后检查 TensorRT 版本，如果打印结果是 8.2.5.1，说明安装 Python 包成功了。

### Linux

和在 Windows 环境下安装类似，默认在一台有 NVIDIA 显卡的机器上，提前安装好 [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) 和 [CUDNN](https://developer.nvidia.com/rdp/cudnn-archive)，登录 NVIDIA 官方网站下载和主机 CUDA 版本适配的 TensorRT 压缩包即可。

以 CUDA 版本是 10.2 为例，选择适配 CUDA 10.2 的 [tar 包](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.5.1/tars/tensorrt-8.2.5.1.linux.x86_64-gnu.cuda-10.2.cudnn8.2.tar.gz)，然后执行类似如下的命令安装并测试：

```shell
cd /the/path/of/tensorrt/tar/gz/file
tar -zxvf TensorRT-8.2.5.1.linux.x86_64-gnu.cuda-10.2.cudnn8.2.tar.gz
export TENSORRT_DIR=$(pwd)/TensorRT-8.2.5.1
export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
pip install TensorRT-8.2.5.1/python/tensorrt-8.2.5.1-cp37-none-linux_x86_64.whl
python -c "import tensorrt;print(tensorrt.__version__)"
```

如果发现打印结果是 8.2.5.1，说明安装 Python 包成功了。

### Jetson

对于 Jetson 平台，我们有非常详细的安装环境配置教程，可参考 [MMDeploy 安装文档](https://github.com/open-mmlab/mmdeploy/tree/main/docs/zh_cn/01-how-to-build/jetsons.md)。需要注意的是，在 Jetson 上配置的 CUDA 版本 TensorRT 版本与 JetPack 强相关的，我们选择适配硬件的版本即可。配置好环境后，通过 `python -c "import tensorrt;print(tensorrt.__version__)"` 查看TensorRT版本是否正确。

## 模型构建

我们使用 TensorRT 生成模型主要有两种方式：

1. 直接通过 TensorRT 的 API 逐层搭建网络；
2. 将中间表示的模型转换成 TensorRT 的模型，比如将 ONNX 模型转换成 TensorRT 模型。

接下来，我们将用 Python 和 C++ 语言分别使用这两种方式构建 TensorRT 模型，并将生成的模型进行推理。

### 直接构建

利用 TensorRT 的 API 逐层搭建网络，这一过程类似使用一般的训练框架，如使用 Pytorch 或者TensorFlow 搭建网络。需要注意的是对于权重部分，如卷积或者归一化层，需要将权重内容赋值到 TensorRT 的网络中。本文就不详细展示，只搭建一个对输入做池化的简单网络。

#### 使用 Python API 构建

首先是使用 Python API 直接搭建 TensorRT 网络，这种方法主要是利用 `tensorrt.Builder` 的 `create_builder_config` 和 `create_network` 功能，分别构建 config 和 network，前者用于设置网络的最大工作空间等参数，后者就是网络主体，需要对其逐层添加内容。

此外，需要定义好输入和输出名称，将构建好的网络序列化，保存成本地文件。值得注意的是：如果想要网络接受不同分辨率的输入输出，需要使用 `tensorrt.Builder` 的 `create_optimization_profile` 函数，并设置最小、最大的尺寸。

实现代码如下：

```python
import tensorrt as trt

verbose = True
IN_NAME = 'input'
OUT_NAME = 'output'
IN_H = 224
IN_W = 224
BATCH_SIZE = 1

EXPLICIT_BATCH = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config(
) as config, builder.create_network(EXPLICIT_BATCH) as network:
    # define network
    input_tensor = network.add_input(
        name=IN_NAME, dtype=trt.float32, shape=(BATCH_SIZE, 3, IN_H, IN_W))
    pool = network.add_pooling(
        input=input_tensor, type=trt.PoolingType.MAX, window_size=(2, 2))
    pool.stride = (2, 2)
    pool.get_output(0).name = OUT_NAME
    network.mark_output(pool.get_output(0))

    # serialize the model to engine file
    profile = builder.create_optimization_profile()
    profile.set_shape_input('input', *[[BATCH_SIZE, 3, IN_H, IN_W]]*3)
    builder.max_batch_size = 1
    config.max_workspace_size = 1 << 30
    engine = builder.build_engine(network, config)
    with open('model_python_trt.engine', mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!")
```

#### 使用 C++ API 构建

对于想要直接用 C++ 语言构建网络的小伙伴来说，整个流程和上述 Python 的执行过程非常类似，需要注意的点主要有：

1. `nvinfer1:: createInferBuilder` 对应 Python 中的 `tensorrt.Builder`，需要传入 `ILogger` 类的实例，但是 `ILogger` 是一个抽象类，需要用户继承该类并实现内部的虚函数。不过此处我们直接使用了 TensorRT 包解压后的 samples 文件夹 ../samples/common/logger.h 文件里的实现 `Logger` 子类。
2. 设置 TensorRT 模型的输入尺寸，需要多次调用 `IOptimizationProfile` 的 `setDimensions` 方法，比 Python `略繁琐一些。IOptimizationProfile` 需要用 `createOptimizationProfile` 函数，对应 Python 的 `create_builder_config` 函数。

实现代码如下：

```cpp
#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <../samples/common/logger.h>

using namespace nvinfer1;
using namespace sample;

const char* IN_NAME = "input";
const char* OUT_NAME = "output";
static const int IN_H = 224;
static const int IN_W = 224;
static const int BATCH_SIZE = 1;
static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

int main(int argc, char** argv)
{
        // Create builder
        Logger m_logger;
        IBuilder* builder = createInferBuilder(m_logger);
        IBuilderConfig* config = builder->createBuilderConfig();

        // Create model to populate the network
        INetworkDefinition* network = builder->createNetworkV2(EXPLICIT_BATCH);
        ITensor* input_tensor = network->addInput(IN_NAME, DataType::kFLOAT, Dims4{ BATCH_SIZE, 3, IN_H, IN_W });
        IPoolingLayer* pool = network->addPoolingNd(*input_tensor, PoolingType::kMAX, DimsHW{ 2, 2 });
        pool->setStrideNd(DimsHW{ 2, 2 });
        pool->getOutput(0)->setName(OUT_NAME);
        network->markOutput(*pool->getOutput(0));

        // Build engine
        IOptimizationProfile* profile = builder->createOptimizationProfile();
        profile->setDimensions(IN_NAME, OptProfileSelector::kMIN, Dims4(BATCH_SIZE, 3, IN_H, IN_W));
        profile->setDimensions(IN_NAME, OptProfileSelector::kOPT, Dims4(BATCH_SIZE, 3, IN_H, IN_W));
        profile->setDimensions(IN_NAME, OptProfileSelector::kMAX, Dims4(BATCH_SIZE, 3, IN_H, IN_W));
        config->setMaxWorkspaceSize(1 << 20);
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

        // Serialize the model to engine file
        IHostMemory* modelStream{ nullptr };
        assert(engine != nullptr);
        modelStream = engine->serialize();

        std::ofstream p("model.engine", std::ios::binary);
        if (!p) {
                std::cerr << "could not open output file to save model" << std::endl;
                return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        std::cout << "generating file done!" << std::endl;

        // Release resources
        modelStream->destroy();
        network->destroy();
        engine->destroy();
        builder->destroy();
        config->destroy();
        return 0;
}
```

### IR 转换模型

除了直接通过 TensorRT 的 API 逐层搭建网络并序列化模型，TensorRT 还支持将中间表示的模型（如 ONNX）转换成 TensorRT 模型。

#### 使用 Python API 转换

我们首先使用 Pytorch 实现一个和上文一致的模型，即只对输入做一次池化并输出；然后将 Pytorch 模型转换成 ONNX 模型；最后将 ONNX 模型转换成 TensorRT 模型。
这里主要使用了 TensorRT 的 `OnnxParser` 功能，它可以将 ONNX 模型解析到 TensorRT 的网络中。最后我们同样可以得到一个 TensorRT 模型，其功能与上述方式实现的模型功能一致。

实现代码如下：

```python
import torch
import onnx
import tensorrt as trt


onnx_model = 'model.onnx'

class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(x)

device = torch.device('cuda:0')

# generate ONNX model
torch.onnx.export(NaiveModel(), torch.randn(1, 3, 224, 224), onnx_model, input_names=['input'], output_names=['output'], opset_version=11)
onnx_model = onnx.load(onnx_model)

# create builder and network
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
EXPLICIT_BATCH = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)

# parse onnx
parser = trt.OnnxParser(network, logger)

if not parser.parse(onnx_model.SerializeToString()):
    error_msgs = ''
    for error in range(parser.num_errors):
        error_msgs += f'{parser.get_error(error)}\n'
    raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

config = builder.create_builder_config()
config.max_workspace_size = 1<<20
profile = builder.create_optimization_profile()

profile.set_shape('input', [1,3 ,224 ,224], [1,3,224, 224], [1,3 ,224 ,224])
config.add_optimization_profile(profile)
# create engine
with torch.cuda.device(device):
    engine = builder.build_engine(network, config)

with open('model.engine', mode='wb') as f:
    f.write(bytearray(engine.serialize()))
    print("generating file done!")
```

IR 转换时，如果有多 Batch、多输入、动态 shape 的需求，都可以通过多次调用 `set_shape` 函数进行设置。`set_shape` 函数接受的传参分别是：输入节点名称，可接受的最小输入尺寸，最优的输入尺寸，可接受的最大输入尺寸。一般要求这三个尺寸的大小关系为单调递增。

#### 使用 C++ API 转换

介绍了如何用 Python 语言将 ONNX 模型转换成 TensorRT 模型后，再介绍下如何用 C++ 将 ONNX 模型转换成 TensorRT 模型。这里通过 `NvOnnxParser`，我们可以将上一小节转换时得到的 ONNX 文件直接解析到网络中。

实现代码如下：

```cpp
#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <../samples/common/logger.h>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace sample;

int main(int argc, char** argv)
{
        // Create builder
        Logger m_logger;
        IBuilder* builder = createInferBuilder(m_logger);
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        IBuilderConfig* config = builder->createBuilderConfig();

        // Create model to populate the network
        INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

        // Parse ONNX file
        IParser* parser = nvonnxparser::createParser(*network, m_logger);
        bool parser_status = parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING));

        // Get the name of network input
        Dims dim = network->getInput(0)->getDimensions();
        if (dim.d[0] == -1)  // -1 means it is a dynamic model
        {
                const char* name = network->getInput(0)->getName();
                IOptimizationProfile* profile = builder->createOptimizationProfile();
                profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
                profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
                profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
                config->addOptimizationProfile(profile);
        }


        // Build engine
        config->setMaxWorkspaceSize(1 << 20);
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

        // Serialize the model to engine file
        IHostMemory* modelStream{ nullptr };
        assert(engine != nullptr);
        modelStream = engine->serialize();

        std::ofstream p("model.engine", std::ios::binary);
        if (!p) {
                std::cerr << "could not open output file to save model" << std::endl;
                return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        std::cout << "generate file success!" << std::endl;

        // Release resources
        modelStream->destroy();
        network->destroy();
        engine->destroy();
        builder->destroy();
        config->destroy();
        return 0;
}
```

### 模型推理

前面，我们使用了两种构建 TensorRT 模型的方式，分别用 Python 和 C++ 两种语言共生成了四个 TensorRT 模型，这四个模型的功能理论上是完全一致的。
接下来，我们将分别使用 Python 和 C++ 两种语言对生成的 TensorRT 模型进行推理。

#### 使用 Python API 推理

首先是使用 Python API 推理 TensorRT 模型，这里部分代码参考了 [MMDeploy](https://github.com/open-mmlab/mmdeploy)。运行下面代码，可以发现输入一个 `1x3x224x224` 的张量，输出一个 `1x3x112x112` 的张量，完全符合我们对输入池化后结果的预期。

```python
from typing import Union, Optional, Sequence,Dict,Any

import torch
import tensorrt as trt

class TRTWrapper(torch.nn.Module):
    def __init__(self,engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        assert self._input_names is not None
        assert self._output_names is not None
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        profile_id = 0
        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            profile = self.engine.get_profile_shape(profile_id, input_name)
            assert input_tensor.dim() == len(
                profile[0]), 'Input dim is different from engine profile.'
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape,
                                             profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            idx = self.engine.get_binding_index(input_name)

            # All input tensors must be gpu variables
            assert 'cuda' in input_tensor.device.type
            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch.float32
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch.device('cuda')
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
        return outputs

model = TRTWrapper('model.engine', ['output'])
output = model(dict(input = torch.randn(1, 3, 224, 224).cuda()))
print(output)
```

#### 使用 C++ API 推理

最后，在很多实际生产环境中，我们都会使用 C++ 语言完成具体的任务，以达到更加高效的代码运行效果，另外 TensoRT 的用户一般也都更看重其在 C++ 下的使用，所以我们也用 C++ 语言实现一遍模型推理，这也可以和用 Python API 推理模型做一个对比。

实现代码如下：

```cpp
#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <../samples/common/logger.h>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;
using namespace sample;

const char* IN_NAME = "input";
const char* OUT_NAME = "output";
static const int IN_H = 224;
static const int IN_W = 224;
static const int BATCH_SIZE = 1;
static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);


void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
        const ICudaEngine& engine = context.getEngine();

        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(engine.getNbBindings() == 2);
        void* buffers[2];

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine.getBindingIndex(IN_NAME);
        const int outputIndex = engine.getBindingIndex(OUT_NAME);

        // Create GPU buffers on device
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * IN_H * IN_W * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 3 * IN_H * IN_W /4 * sizeof(float)));

        // Create stream
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 3 * IN_H * IN_W / 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
        // create a model using the API directly and serialize it to a stream
        char *trtModelStream{ nullptr };
        size_t size{ 0 };

        std::ifstream file("model.engine", std::ios::binary);
        if (file.good()) {
                file.seekg(0, file.end);
                size = file.tellg();
                file.seekg(0, file.beg);
                trtModelStream = new char[size];
                assert(trtModelStream);
                file.read(trtModelStream, size);
                file.close();
        }

        Logger m_logger;
        IRuntime* runtime = createInferRuntime(m_logger);
        assert(runtime != nullptr);
        ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
        assert(engine != nullptr);
        IExecutionContext* context = engine->createExecutionContext();
        assert(context != nullptr);

        // generate input data
        float data[BATCH_SIZE * 3 * IN_H * IN_W];
        for (int i = 0; i < BATCH_SIZE * 3 * IN_H * IN_W; i++)
                data[i] = 1;

        // Run inference
        float prob[BATCH_SIZE * 3 * IN_H * IN_W /4];
        doInference(*context, data, prob, BATCH_SIZE);

        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();
        return 0;
}
```

## 总结

通过本文的学习，我们掌握了两种构建 TensorRT 模型的方式：直接通过 TensorRT 的 API 逐层搭建网络；将中间表示的模型转换成 TensorRT 的模型。不仅如此，我们还分别用 C++ 和 Python 两种语言完成了 TensorRT 模型的构建及推理，相信大家都有所收获！在下一篇文章中，我们将和大家一起学习何添加 TensorRT 自定义算子，敬请期待哦~

## FAQ

1. Could not find: cudnn64_8.dll. Is it on your PATH?
   首先检查下自己的环境变量中是否包含 cudnn64_8.dll 所在的路径，若发现 cudnn 的路径在 C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\bin 中，但是里面只有 cudnn64_7.dll。解决方法是去 NVIDIA 官网下载 cuDNN zip 包，解压后，复制其中的 cudnn64_8.dll 到 CUDA Toolkit 的 bin 目录下。这时也可以复制一份 cudnn64_7.dll，然后将复制的那份改名成 cudnn64_8.dll，同样可以解决这个问题。
