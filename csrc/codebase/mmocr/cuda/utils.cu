// Copyright (c) OpenMMLab. All rights reserved.

#include "codebase/mmocr/cuda/utils.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/transform.h"

namespace mmdeploy::mmocr {

namespace dbnet {

struct _op {
  const float* data;
  float* score;
  uint8_t* mask;
  float thr;
  __device__ void operator()(int idx) const {
    float v = 1.f / (1.f + expf(-data[idx]));
    score[idx] = v;
    mask[idx] = v >= thr;
  }
};

void SigmoidAndThreshold(const float* d_data, int n, float thr, float* d_score, uint8_t* d_mask) {
  thrust::counting_iterator<int> index;

  thrust::for_each_n(index, n, _op{d_data, d_score, d_mask, thr});
}

}  // namespace dbnet

}  // namespace mmdeploy::mmocr
