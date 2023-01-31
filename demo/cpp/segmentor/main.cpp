// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/segmentor.hpp"

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <random>
#include <string>
#include <vector>

using namespace std;

vector<cv::Vec3b> gen_palette(int num_classes) {
  std::mt19937 gen;
  std::uniform_int_distribution<ushort> uniform_dist(0, 255);

  vector<cv::Vec3b> palette;
  palette.reserve(num_classes);
  for (auto i = 0; i < num_classes; ++i) {
    palette.emplace_back(uniform_dist(gen), uniform_dist(gen), uniform_dist(gen));
  }
  return palette;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage:" << std::endl
              << "  ./segmentor device_name sdk_model_path "
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

  mmdeploy::Segmentor segmentor{mmdeploy::Model{model_path}, context};

  cv::Mat img = cv::imread(image_path);
  if (!img.data) {
    fprintf(stderr, "failed to load image: %s\n", image_path);
    return 1;
  }
  auto result = segmentor.Apply(img);

  auto palette = gen_palette(result->classes + 1);

  cv::Mat color_mask = cv::Mat::zeros(result->height, result->width, CV_8UC3);
  int pos = 0;
  for (auto iter = color_mask.begin<cv::Vec3b>(); iter != color_mask.end<cv::Vec3b>(); ++iter) {
    *iter = palette[result->mask[pos++]];
  }

  img = img * 0.5 + color_mask * 0.5;
  cv::imwrite("output_segmentation.png", img);

  return 0;
}
