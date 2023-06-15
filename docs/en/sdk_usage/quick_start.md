# Quick Start

In terms of model deployment, most ML models require some preprocessing steps on the input data and postprocessing steps on the output to get structured output. MMDeploy sdk provides a lot of pre-processing and post-processing process. When you convert and deploy a model, you can enjoy the convenience brought by mmdeploy sdk.

## Model Conversion

You can refer to [convert model](../02-how-to-run/convert_model.md) for more details.

After model conversion with `--dump-info`, the structure of model directory (tensorrt model) is as follows. If you convert to other backend, the structure will be slightly different. The two images are for quick conversion validation.

```bash
├── deploy.json
├── detail.json
├── pipeline.json
├── end2end.onnx
├── end2end.engine
├── output_pytorch.jpg
└── output_tensorrt.jpg
```

The files related to sdk are:

- deploy.json    // model information.
- pipeline.json  // inference information.
- end2end.engine // model file for tensort, will be different for other backends.

SDK can read the model directory directly or you can pack the related files to zip archive for better distribution or encryption. To read the zip file, the sdk should build with `-DMMDEPLOY_ZIP_MODEL=ON`

## SDK Inference

Generally speaking, there are three steps to inference a model.

- Create a pipeline
- Load the data
- Model inference

We use `classifier` as an example to show these three steps.

### Create a pipeline

#### Load model from disk

```cpp

std::string model_path = "/data/resnet"; // or "/data/resnet.zip" if build with `-DMMDEPLOY_ZIP_MODEL=ON`
mmdeploy_model_t model;
mmdeploy_model_create_by_path(model_path, &model);

mmdeploy_classifier_t classifier{};
mmdeploy_classifier_create(model, "cpu", 0, &classifier);
```

#### Load model from memory

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

### Load the data

```cpp
cv::Mat img = cv::imread(image_path);
```

### Model inference

```cpp
mmdeploy_classification_t* res{};
int* res_count{};
mmdeploy_classifier_apply(classifier, &mat, 1, &res, &res_count);
```
