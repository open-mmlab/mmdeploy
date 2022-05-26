// Copyright (c) OpenMMLab. All rights reserved.

#include "codebase/mmocr/cuda/utils.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/transform.h"

namespace mmdeploy::mmocr {

namespace panet {

__device__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

struct _process_masks_op {
  const float* text_pred;
  const float* kernel_pred;
  float text_thr;
  float kernel_thr;
  uint8_t* text_mask;
  uint8_t* kernel_mask;
  float* text_score;
  __device__ void operator()(int index) const {
    auto text_sigmoid = sigmoid(text_pred[index]);
    auto kernel_sigmoid = sigmoid(kernel_pred[index]);
    text_score[index] = text_sigmoid;
    auto text_valid = text_sigmoid > text_thr;
    text_mask[index] = text_valid ? 255 : 0;
    kernel_mask[index] = (text_valid && kernel_sigmoid > kernel_thr) ? 255 : 0;
  }
};

void ProcessMasks(const float* d_text_pred, const float* d_kernel_pred, float text_thr,
                  float kernel_thr, int n, uint8_t* d_text_mask, uint8_t* d_kernel_mask,
                  float* d_text_score, cudaStream_t stream) {
  thrust::counting_iterator<int> index{0};
  _process_masks_op op{d_text_pred, d_kernel_pred, text_thr,    kernel_thr,
                       d_text_mask, d_kernel_mask, d_text_score};
  thrust::for_each_n(thrust::cuda::par.on(stream), index, n, op);
}

struct _transpose_op {
  const float* input;
  float* output;
  int h;
  int w;
  __device__ void operator()(int index) const {
    int i = index / w;
    int j = index % w;
    output[j * h + i] = input[index];
  }
};

void Transpose(const float* d_input, int h, int w, float* d_output, cudaStream_t stream) {
  thrust::counting_iterator<int> index{0};
  _transpose_op op{d_input, d_output, h, w};
  thrust::for_each_n(thrust::cuda::par.on(stream), index, h * w, op);
}

}  // namespace panet

namespace dbnet {

struct _threshold_op {
  float thr;
  __device__ bool operator()(float score) const { return score >= thr; }
};

void Threshold(const float* d_score, int n, float thr, uint8_t* d_mask, cudaStream_t stream) {
  _threshold_op op{thr};
  thrust::transform(thrust::cuda::par.on(stream), d_score, d_score + n, d_mask, op);
}

}  // namespace dbnet

}  // namespace mmdeploy::mmocr
