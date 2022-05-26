// Copyright (c) OpenMMLab. All rights reserved.

#include "codebase/mmocr/cuda/utils.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/transform.h"

namespace mmdeploy::mmocr {

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
