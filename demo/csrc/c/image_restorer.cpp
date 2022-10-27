// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "mmdeploy/restorer.h"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "usage:\n  image_restorer device_name model_path image_path\n");
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

  mmdeploy_restorer_t restorer{};
  int status{};
  status = mmdeploy_restorer_create_by_path(model_path, device_name, 0, &restorer);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create restorer, code: %d\n", (int)status);
    return 1;
  }

  mmdeploy_mat_t mat{
      img.data, img.rows, img.cols, 3, MMDEPLOY_PIXEL_FORMAT_BGR, MMDEPLOY_DATA_TYPE_UINT8};

  mmdeploy_mat_t* result{};
  status = mmdeploy_restorer_apply(restorer, &mat, 1, &result);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to apply restorer, code: %d\n", (int)status);
    return 1;
  }

  cv::Mat sr_img(result->height, result->width, CV_8UC3, result->data);
  cv::cvtColor(sr_img, sr_img, cv::COLOR_RGB2BGR);
  cv::imwrite("output_restorer.bmp", sr_img);

  mmdeploy_restorer_release_result(result, 1);
  mmdeploy_restorer_destroy(restorer);

  return 0;
}
