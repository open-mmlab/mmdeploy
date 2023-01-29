// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>
#include <numeric>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <random>
#include <string>
#include <vector>

#include "mmdeploy/segmentor.h"

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

  mmdeploy_segmentor_t segmentor{};
  int status{};
  status = mmdeploy_segmentor_create_by_path(model_path, device_name, 0, &segmentor);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create segmentor, code: %d\n", (int)status);
    return 1;
  }

  mmdeploy_mat_t mat{
      img.data, img.rows, img.cols, 3, MMDEPLOY_PIXEL_FORMAT_BGR, MMDEPLOY_DATA_TYPE_UINT8};

  mmdeploy_segmentation_t* result{};
  status = mmdeploy_segmentor_apply(segmentor, &mat, 1, &result);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to apply segmentor, code: %d\n", (int)status);
    return 1;
  }

  auto palette = gen_palette(result->classes + 1);

  cv::Mat color_mask = cv::Mat::zeros(result->height, result->width, CV_8UC3);
  int pos = 0;
  int total = color_mask.rows * color_mask.cols;
  std::vector<int> idxs(result->classes);
  for (auto iter = color_mask.begin<cv::Vec3b>(); iter != color_mask.end<cv::Vec3b>(); ++iter) {
    // output mask
    if (result->mask) {
      *iter = palette[result->mask[pos++]];
    }
    // output score
    if (result->score) {
      std::iota(idxs.begin(), idxs.end(), 0);
      auto k =
          std::max_element(idxs.begin(), idxs.end(),
                           [&](int i, int j) {
                             return result->score[i * total + pos] < result->score[j * total + pos];
                           }) -
          idxs.begin();
      *iter = palette[k];
      pos += 1;
    }
  }

  img = img * 0.5 + color_mask * 0.5;
  cv::imwrite("output_segmentation.png", img);

  mmdeploy_segmentor_release_result(result, 1);
  mmdeploy_segmentor_destroy(segmentor);

  return 0;
}
