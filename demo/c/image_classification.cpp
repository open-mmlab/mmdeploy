#include <fstream>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <string>

#include "mmdeploy/classifier.h"

int main(int argc, char* argv[]) {
  if (argc < 4) {
    fprintf(stderr,
            "usage:\n  ./image_classification device_name sdk_model_path image_path [--profile]\n");
    return 1;
  }
  auto device_name = argv[1];
  auto model_path = argv[2];
  auto image_path = argv[3];
  auto profile = argc > 4 ? std::string("--profile") == argv[argc - 1] : false;

  int status{};
  mmdeploy_context_t context{};
  mmdeploy_device_t device{};
  mmdeploy_model_t model{};
  mmdeploy_profiler_t profiler{};
  mmdeploy_classifier_t classifier{};


  if ((status = mmdeploy_context_create(&context)) != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create context, code: %d\n", status);
    return 1;
  }

  if ((status = mmdeploy_device_create(device_name, 0, &device)) != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create device, code: %d\n", status);
    return 1;
  }

  if ((status = mmdeploy_profiler_create("profiler.bin", &profiler)) != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create profiler, code: %d\n", status);
    return 1;
  }

  mmdeploy_context_add(context, MMDEPLOY_TYPE_DEVICE, device_name, device);
  mmdeploy_context_add(context, MMDEPLOY_TYPE_PROFILER, "", profiler);
  if ((status = mmdeploy_model_create_by_path(model_path, &model)) != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create model, code: %d\n", status);
    return 1;
  }

  status = mmdeploy_classifier_create_v2(model, context, &classifier);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create classifier, code: %d\n", status);
    return 1;
  }

  cv::Mat img = cv::imread(image_path);
  if (!img.data) {
    fprintf(stderr, "failed to load image: %s\n", image_path);
    return 1;
  }
  mmdeploy_mat_t mat{
      img.data, img.rows, img.cols, 3, MMDEPLOY_PIXEL_FORMAT_BGR, MMDEPLOY_DATA_TYPE_UINT8};

  mmdeploy_classification_t* res{};
  int* res_count{};
  status = mmdeploy_classifier_apply(classifier, &mat, 1, &res, &res_count);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to apply classifier, code: %d\n", status);
    return 1;
  }
  for (int i = 0; i < res_count[0]; ++i) {
    fprintf(stdout, "label: %d, score: %.4f\n", res[i].label_id, res[i].score);
  }

  mmdeploy_classifier_release_result(res, res_count, 1);

  mmdeploy_profiler_destroy(profiler);
  mmdeploy_device_destroy(device);
  mmdeploy_model_destroy(model);
  mmdeploy_context_destroy(context);
  mmdeploy_classifier_destroy(classifier);
  return 0;
}
