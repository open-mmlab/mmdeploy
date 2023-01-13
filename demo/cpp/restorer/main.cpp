// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/restorer.hpp"

#include <iostream>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage:" << std::endl
              << "  ./restorer device_name sdk_model_path "
                 "image_path [--profile]"
              << std::endl;
    return 1;
  }
  auto device_name = argv[1];
  auto model_path = argv[2];
  auto image_path = argv[3];
  auto profile = argc > 4 ? std::string("--profile") == argv[argc - 1] : false;

  mmdeploy::Context context(mmdeploy::Device{device_name});
  mmdeploy::Profiler profiler("profiler.bin");
  if (profile) {
    context.Add(profiler);
  }

  mmdeploy::Restorer restorer{mmdeploy::Model{model_path}, context};

  cv::Mat img = cv::imread(image_path);
  if (!img.data) {
    std::cerr << "failed to load image: " << image_path << std::endl;
    return 1;
  }
  auto result = restorer.Apply(img);

  cv::Mat sr_img(result->height, result->width, CV_8UC3, result->data);
  cv::cvtColor(sr_img, sr_img, cv::COLOR_RGB2BGR);
  cv::imwrite("output_restorer.bmp", sr_img);

  return 0;
}
