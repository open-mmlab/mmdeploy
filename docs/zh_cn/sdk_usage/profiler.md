# Pipeline 速度分析 (profiler)

sdk 提供 pipeline 各模块耗时统计功能，默认关闭，若要使用该功能，需要两个步骤：

- 生成性能数据
- 分析性能数据

## 生成性能数据

以 C 接口，分类 pipeline 为例。在创建 pipeline 时需要使用带有 context 信息的接口，并在 context 中加入 profiler 信息。 详细代码如下。 正常运行 demo 会在当前目录生成 profiler 数据 `profiler_data.txt`。

```c++
#include <fstream>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <string>

#include "mmdeploy/classifier.h"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "usage:\n  image_classification device_name dump_model_directory image_path\n");
    return 1;
  }
  auto device_name = argv[1];
  auto model_path = argv[2];
  auto image_path = argv[3];
  cv::Mat img = cv::imread(image_path);
  if (!img.data) {
    fprintf(stderr, "failed to load image: %s\n", image_path);
    return 1;
  }

  mmdeploy_model_t model{};
  mmdeploy_model_create_by_path(model_path, &model);

  // create profiler and add it to context
  // profiler data will save to profiler_data.txt
  mmdeploy_profiler_t profiler{};
  mmdeploy_profiler_create("profiler_data.txt", &profiler);

  mmdeploy_context_t context{};
  mmdeploy_context_create_by_device(device_name, 0, &context);
  mmdeploy_context_add(context, MMDEPLOY_TYPE_PROFILER, nullptr, profiler);

  mmdeploy_classifier_t classifier{};
  int status{};
  status = mmdeploy_classifier_create_v2(model, context, &classifier);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create classifier, code: %d\n", (int)status);
    return 1;
  }

  mmdeploy_mat_t mat{
      img.data, img.rows, img.cols, 3, MMDEPLOY_PIXEL_FORMAT_BGR, MMDEPLOY_DATA_TYPE_UINT8};

  // inference loop
  for (int i = 0; i < 100; i++) {
    mmdeploy_classification_t* res{};
    int* res_count{};
    status = mmdeploy_classifier_apply(classifier, &mat, 1, &res, &res_count);

    mmdeploy_classifier_release_result(res, res_count, 1);
  }

  mmdeploy_classifier_destroy(classifier);

  mmdeploy_model_destroy(model);
  mmdeploy_profiler_destroy(profiler);
  mmdeploy_context_destroy(context);

  return 0;
}

```

## 分析性能数据

使用脚本可对性能数据进行解析。

```bash
python tools/sdk_analyze.py profiler_data.txt
```

解析结果如下，其中 name 表示节点的名称，n_call表示调用的次数，t_mean 表示平均耗时，t_50% t_90% 表示耗时的百分位数。

```bash
+---------------------------+--------+-------+--------+--------+-------+-------+
|           name            | occupy | usage | n_call | t_mean | t_50% | t_90% |
+===========================+========+=======+========+========+=======+=======+
| ./Pipeline                | -      | -     | 100    | 4.831  | 1.913 | 1.946 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|     Preprocess/Compose    | -      | -     | 100    | 0.125  | 0.118 | 0.144 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|         LoadImageFromFile | 0.017  | 0.017 | 100    | 0.081  | 0.077 | 0.098 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|         Resize            | 0.003  | 0.003 | 100    | 0.012  | 0.012 | 0.013 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|         CenterCrop        | 0.002  | 0.002 | 100    | 0.008  | 0.008 | 0.008 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|         Normalize         | 0.002  | 0.002 | 100    | 0.009  | 0.009 | 0.009 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|         ImageToTensor     | 0.002  | 0.002 | 100    | 0.008  | 0.007 | 0.007 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|         Collect           | 0.001  | 0.001 | 100    | 0.005  | 0.005 | 0.005 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|     resnet                | 0.968  | 0.968 | 100    | 4.678  | 1.767 | 1.774 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|     postprocess           | 0.003  | 0.003 | 100    | 0.015  | 0.015 | 0.017 |
+---------------------------+--------+-------+--------+--------+-------+-------+
```
