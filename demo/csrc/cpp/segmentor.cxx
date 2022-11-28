// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/segmentor.hpp"

#include <fstream>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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
  if (argc != 4) {
    fprintf(stderr, "usage:\n  image_segmentation device_name model_path image_path\n");
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

  Segmentor segmentor{Model{model_path}, Device{device_name}};

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
