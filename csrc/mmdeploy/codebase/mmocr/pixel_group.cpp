// Copyright (c) OpenMMLab. All rights reserved.
// Modified from https://github.com/WenmuZhou/PAN.pytorch
// and
// https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/cpu/pixel_group.cpp

#include <cmath>
#include <queue>
#include <vector>

#include "mmdeploy/core/tensor.h"
#include "opencv2/imgproc/imgproc.hpp"

namespace mmdeploy::mmocr {

std::vector<std::vector<float>> estimate_confidence(const int32_t* label, const float* score,
                                                    int label_num, int height, int width) {
  std::vector<std::vector<float>> point_vector;
  for (int i = 0; i < label_num; i++) {
    std::vector<float> point;
    point.push_back(0);
    point.push_back(0);
    point_vector.push_back(point);
  }
  for (int y = 0; y < height; y++) {
    auto label_tmp = label + y * width;
    auto score_tmp = score + y * width;
    for (int x = 0; x < width; x++) {
      auto l = label_tmp[x];
      if (l > 0) {
        float confidence = score_tmp[x];
        point_vector[l].push_back(x);
        point_vector[l].push_back(y);
        point_vector[l][0] += confidence;
        point_vector[l][1] += 1;
      }
    }
  }
  for (size_t l = 0; l < point_vector.size(); l++)
    if (point_vector[l][1] > 0) {
      point_vector[l][0] /= point_vector[l][1];
    }
  return point_vector;
}

std::vector<std::vector<float>> pixel_group_cpu(const cv::Mat_<float>& score,
                                                const cv::Mat_<uint8_t>& mask,
                                                const cv::Mat_<float>& embedding,
                                                const cv::Mat_<int32_t>& kernel_label,
                                                const cv::Mat_<uint8_t>& kernel_contour,
                                                int kernel_region_num, float dis_threshold) {
  int height = score.rows;
  int width = score.cols;
  assert(embedding.rows == height * width);
  assert(height == mask.rows);
  assert(width == mask.cols);

  auto threshold_square = dis_threshold * dis_threshold;
  auto ptr_score = score.ptr<float>();
  auto ptr_mask = mask.ptr<uint8_t>();
  auto ptr_kernel_contour = kernel_contour.ptr<uint8_t>();
  auto ptr_embedding = embedding.ptr<float>();
  auto ptr_kernel_label = kernel_label.ptr<int32_t>();
  std::queue<std::tuple<int, int, int32_t>> contour_pixels;
  auto embedding_dim = embedding.cols;
  std::vector<std::vector<float>> kernel_vector(kernel_region_num,
                                                std::vector<float>(embedding_dim + 1, 0));

  cv::Mat_<int32_t> text_label = kernel_label.clone();
  auto ptr_text_label = text_label.ptr<int32_t>();

  for (int i = 0; i < height; i++) {
    auto ptr_embedding_tmp = ptr_embedding + i * width * embedding_dim;
    auto ptr_kernel_label_tmp = ptr_kernel_label + i * width;
    auto ptr_kernel_contour_tmp = ptr_kernel_contour + i * width;

    for (int j = 0, k = 0; j < width && k < width * embedding_dim; j++, k += embedding_dim) {
      int32_t label = ptr_kernel_label_tmp[j];
      if (label > 0) {
        for (int d = 0; d < embedding_dim; d++) kernel_vector[label][d] += ptr_embedding_tmp[k + d];
        kernel_vector[label][embedding_dim] += 1;
        // kernel pixel number
        if (ptr_kernel_contour_tmp[j]) {
          contour_pixels.push(std::make_tuple(i, j, label));
        }
      }
    }
  }
  for (int i = 0; i < kernel_region_num; i++) {
    for (int j = 0; j < embedding_dim; j++) {
      kernel_vector[i][j] /= kernel_vector[i][embedding_dim];
    }
  }
  int dx[4] = {-1, 1, 0, 0};
  int dy[4] = {0, 0, -1, 1};
  while (!contour_pixels.empty()) {
    auto query_pixel = contour_pixels.front();
    contour_pixels.pop();
    int y = std::get<0>(query_pixel);
    int x = std::get<1>(query_pixel);
    int32_t l = std::get<2>(query_pixel);
    auto kernel_cv = kernel_vector[l];
    for (int idx = 0; idx < 4; idx++) {
      int tmpy = y + dy[idx];
      int tmpx = x + dx[idx];
      auto ptr_text_label_tmp = ptr_text_label + tmpy * width;
      if (tmpy < 0 || tmpy >= height || tmpx < 0 || tmpx >= width) continue;
      if (!ptr_mask[tmpy * width + tmpx] || ptr_text_label_tmp[tmpx] > 0) continue;

      float dis = 0;
      auto ptr_embedding_tmp = ptr_embedding + tmpy * width * embedding_dim;
      for (size_t i = 0; i < embedding_dim; i++) {
        dis += std::pow(kernel_cv[i] - ptr_embedding_tmp[tmpx * embedding_dim + i], 2);
        // ignore further computing if dis is big enough
        if (dis >= threshold_square) break;
      }
      if (dis >= threshold_square) continue;
      contour_pixels.push(std::make_tuple(tmpy, tmpx, l));
      ptr_text_label_tmp[tmpx] = l;
    }
  }

  return estimate_confidence(ptr_text_label, ptr_score, kernel_region_num, height, width);
}

}  // namespace mmdeploy::mmocr
