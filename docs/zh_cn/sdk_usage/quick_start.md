# 快速开始

大多数 ML 模型除了推理外，需要对输入数据进行一些预处理，并对输出进行一些后处理步骤，以获得结构化输出。 MMDeploy sdk 提供了常见的预处理和后处理步骤。 当您使用 MMDeploy 进行模型转换后，您可以直接使用mmdeploy sdk 进行推理。

## 模型转换

可参考 [convert model](../02-how-to-run/convert_model.md) 获得更多信息.

转模型时通过增加 `--dump-info` 参数得到如下的目录结构(tensorrt)。 如果转换为其他后端，结构会略有不同。其中两个图片为不同后端推理结果

```bash
├── deploy.json
├── detail.json
├── pipeline.json
├── end2end.onnx
├── end2end.engine
├── output_pytorch.jpg
└── output_tensorrt.jpg
```

和SDK相关的文件有：

- deploy.json    // 模型信息.
- pipeline.json  // pipeline信息，包括前处理、模型以及后处理.
- end2end.engine // 模型文件

SDK 可以直接读取模型目录，也可以读取相关文件打包成 zip 压缩包。 要读取 zip 文件，sdk 在编译时要设置 `-DMMDEPLOY_ZIP_MODEL=ON`

## SDK 推理

一般来讲，模型推理包含以下三个部分。

- 创建 pipeline
- 读取数据
- 模型推理

以下使用 `classifier` 作为例子来展示三个步骤.

### 创建 pipeline

#### 从硬盘中加载模型

```cpp

std::string model_path = "/data/resnet"; // or "/data/resnet.zip" if build with `-DMMDEPLOY_ZIP_MODEL=ON`
mmdeploy_model_t model;
mmdeploy_model_create_by_path(model_path, &model);

mmdeploy_classifier_t classifier{};
mmdeploy_classifier_create(model, "cpu", 0, &classifier);
```

#### 从内存中加载模型

```cpp
std::string model_path = "/data/resnet.zip"
std::ifstream ifs(model_path, std::ios::binary); // /path/to/zipmodel
ifs.seekg(0, std::ios::end);
auto size = ifs.tellg();
ifs.seekg(0, std::ios::beg);
std::string str(size, '\0'); // binary data, should decrypt if it's encrypted
ifs.read(str.data(), size);

mmdeploy_model_t model;
mmdeploy_model_create(str.data(), size, &model);

mmdeploy_classifier_t classifier{};
mmdeploy_classifier_create(model, "cpu", 0, &classifier);
```

### 读取数据

```cpp
cv::Mat img = cv::imread(image_path);
```

### 模型推理

```cpp
mmdeploy_classification_t* res{};
int* res_count{};
mmdeploy_classifier_apply(classifier, &mat, 1, &res, &res_count);
```
