# profiler

The SDK has ability to record the time consumption of each module in the pipeline. It's closed by default. To use this ability, two steps are required:

- Generate profiler data
- Analyze profiler Data

## Generate profiler data

Using the C interface and classification pipeline as an example, when creating the pipeline, the create api with context information needs to be used, and profiler handle needs to be added to the context. The detailed code is shown below. Running the demo normally will generate profiler data "profiler_data.txt" in the current directory.

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

## Analyze profiler Data

The performance data can be visualized using a script.

```bash
python tools/sdk_analyze.py profiler_data.txt
```

The parsing results are as follows: "name" represents the name of the node, "n_call" represents the number of calls, "t_mean" represents the average time consumption, "t_50%" and "t_90%" represent the percentiles of the time consumption.

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
