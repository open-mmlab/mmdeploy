// Copyright (c) OpenMMLab. All rights reserved

#ifndef FAST_CC__CONNECTED_COMPONENT_H_
#define FAST_CC__CONNECTED_COMPONENT_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "opencv2/core.hpp"

namespace mmdeploy {

class ConnectedComponents {
 public:
  explicit ConnectedComponents(void* stream);

  ~ConnectedComponents();

  void Resize(int height, int width);

  int GetComponents(const uint8_t* d_mask, int* h_label);

  void GetContours(std::vector<std::vector<cv::Point>>& corners);

  void GetStats(const uint8_t* d_mask, const float* d_score, std::vector<float>& scores,
                std::vector<int>& areas);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mmdeploy

#endif  // FAST_CC__CONNECTED_COMPONENT_H_
