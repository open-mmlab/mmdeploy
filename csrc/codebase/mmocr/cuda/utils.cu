// Copyright (c) OpenMMLab. All rights reserved.

#include "codebase/mmocr/cuda/utils.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/transform.h"

namespace mmdeploy::mmocr {

namespace panet {

struct _op {
  const float* logit;
  float* score;
  uint8_t* mask;
  float thr;
  __device__ void operator()(int index) const {
    float sigmoid = 1.f / (1.f + expf(-logit[index]));
    if (score) {
      score[index] = sigmoid;
    }
    mask[index] = sigmoid >= thr;
  }
};

void SigmoidAndThreshold(const float* d_logit, int n, float thr, uint8_t* d_mask, float* d_score,
                         cudaStream_t stream) {
  thrust::counting_iterator<int> index{0};
  thrust::for_each_n(thrust::cuda::par.on(stream), index, n, _op{d_logit, d_score, d_mask, thr});
}

}  // namespace panet

namespace dbnet {

struct _op {
  float thr;
  __device__ bool operator()(float score) const { return score >= thr; }
};

void Threshold(const float* d_score, int n, float thr, uint8_t* d_mask, cudaStream_t stream) {
  thrust::transform(thrust::cuda::par.on(stream), d_score, d_score + n, d_mask, _op{thr});
}

}  // namespace dbnet

}  // namespace mmdeploy::mmocr
