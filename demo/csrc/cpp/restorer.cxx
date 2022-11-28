// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/restorer.hpp"

#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

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

  using namespace mmdeploy;

  Restorer restorer{Model{model_path}, Device{device_name}};

  auto result = restorer.Apply(img);

  cv::Mat sr_img(result->height, result->width, CV_8UC3, result->data);
  cv::cvtColor(sr_img, sr_img, cv::COLOR_RGB2BGR);
  cv::imwrite("output_restorer.bmp", sr_img);

  return 0;
}
