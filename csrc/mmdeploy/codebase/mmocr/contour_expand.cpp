// Copyright (c) OpenMMLab. All rights reserved
// Modified from https://github.com/whai362/PSENet
// and
// https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/contour_expand.cpp

#include <cmath>
#include <iostream>
#include <queue>
#include <vector>

#include "mmdeploy/core/tensor.h"
#include "opencv2/opencv.hpp"

namespace mmdeploy::mmocr {

using namespace std;
using cv::Mat_;

class Point2d {
 public:
  int x;
  int y;

  Point2d() : x(0), y(0) {}
  Point2d(int _x, int _y) : x(_x), y(_y) {}
};

void kernel_dilate(const uint8_t* data, int kernel_num, int height, int width, const int* label_map,
                   int label_num, const float* score_map, int min_area, Mat_<int32_t>& text_labels,
                   vector<int>& text_areas, vector<float>& text_scores,
                   vector<vector<int>>& text_points) {
  text_labels = Mat_<int32_t>::zeros(height, width);
  text_areas = vector<int>(label_num);
  text_scores = vector<float>(label_num);
  text_points = vector<vector<int>>(label_num);

  for (int x = 0; x < height; ++x) {
    for (int y = 0; y < width; ++y) {
      int label = label_map[x * width + y];
      if (label == 0) continue;
      text_areas[label] += 1;
      text_scores[label] += score_map[x * width + y];
      text_points[label].push_back(y);
      text_points[label].push_back(x);
    }
  }

  queue<Point2d> queue, next_queue;
  for (int x = 0; x < height; ++x) {
    auto row = text_labels[x];
    for (int y = 0; y < width; ++y) {
      int label = label_map[x * width + y];
      if (label == 0) continue;
      if (text_areas[label] < min_area) continue;
      Point2d point(x, y);
      queue.push(point);
      row[y] = label;
    }
  }

  const int dx[] = {-1, 1, 0, 0};
  const int dy[] = {0, 0, -1, 1};
  vector<int> kernel_step(kernel_num);
  std::for_each(kernel_step.begin(), kernel_step.end(), [=](int& k) { return k * height * width; });

  for (int kernel_id = kernel_num - 2; kernel_id >= 0; --kernel_id) {
    while (!queue.empty()) {
      Point2d point = queue.front();
      queue.pop();
      int x = point.x;
      int y = point.y;
      int label = text_labels[x][y];
      bool is_edge = true;
      for (int d = 0; d < 4; ++d) {
        int tmp_x = x + dx[d];
        int tmp_y = y + dy[d];
        if (tmp_x < 0 || tmp_x >= height) continue;
        if (tmp_y < 0 || tmp_y >= width) continue;
        int kernel_value = data[kernel_step[kernel_id] + tmp_x * width + tmp_y];
        if (kernel_value == 0) continue;
        if (text_labels[tmp_x][tmp_y] > 0) continue;
        Point2d point(tmp_x, tmp_y);
        queue.push(point);
        text_labels[tmp_x][tmp_y] = label;
        text_areas[label] += 1;
        text_scores[label] += score_map[tmp_x * width + tmp_y];
        text_points[label].push_back(tmp_y);
        text_points[label].push_back(tmp_x);
        is_edge = false;
      }
      if (is_edge) {
        next_queue.push(point);
      }
    }
    swap(queue, next_queue);
  }

  for (int i = 1; i < label_num; ++i) {
    if (text_areas[i]) {
      text_scores[i] /= static_cast<float>(text_areas[i]);
    }
  }
}

void contour_expand(const Mat_<uint8_t>& kernel_masks, const Mat_<int32_t>& kernel_label,
                    const Mat_<float>& score, int min_kernel_area, int kernel_num,
                    vector<int>& text_areas, vector<float>& text_scores,
                    vector<vector<int>>& text_points) {
  assert(kernel_masks.cols == kernel_label.total());
  assert(score.size() == kernel_label.size());

  auto ptr_data = kernel_masks.ptr<uint8_t>();
  auto data_score_map = score.ptr<float>();
  auto data_label_map = kernel_label.ptr<int32_t>();
  vector<vector<int>> text_line;

  Mat_<int32_t> text_labels;

  kernel_dilate(ptr_data, kernel_masks.rows, kernel_label.rows, kernel_label.cols, data_label_map,
                kernel_num, data_score_map, min_kernel_area, text_labels, text_areas, text_scores,
                text_points);
}

}  // namespace mmdeploy::mmocr
