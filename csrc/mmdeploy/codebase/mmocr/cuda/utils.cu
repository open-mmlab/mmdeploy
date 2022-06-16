// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmocr/cuda/utils.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/transform.h"

namespace mmdeploy {

namespace mmocr {

__device__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

namespace panet {

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
  thrust::for_each_n(thrust::cuda::par.on(stream), thrust::counting_iterator<int>(0), n,
                     _process_masks_op{d_text_pred, d_kernel_pred, text_thr, kernel_thr,
                                       d_text_mask, d_kernel_mask, d_text_score});
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
  thrust::for_each_n(thrust::cuda::par.on(stream), thrust::counting_iterator<int>(0), h * w,
                     _transpose_op{d_input, d_output, h, w});
}

}  // namespace panet

namespace dbnet {

struct _threshold_op {
  float thr;
  __device__ bool operator()(float score) const { return score >= thr; }
};

void Threshold(const float* d_score, int n, float thr, uint8_t* d_mask, cudaStream_t stream) {
  thrust::transform(thrust::cuda::par.on(stream), d_score, d_score + n, d_mask, _threshold_op{thr});
}

}  // namespace dbnet

namespace psenet {

struct _process_masks_op {
  const float* preds;
  int c;
  int n;
  float thr;
  uint8_t* masks;
  float* score;
  __device__ void operator()(int index) const {
    bool m0 = false;
    for (int i = 0; i < c; ++i) {
      auto v = sigmoid(preds[i * n + index]);
      if (i == 0) {
        score[index] = v;
        m0 = v > thr;
      }
      masks[i * n + index] = (m0 && v > thr) ? 255 : 0;
    }
  }
};

void ProcessMasks(const float* d_preds, int c, int n, float thr, uint8_t* d_masks, float* d_score,
                  cudaStream_t stream) {
  thrust::for_each_n(thrust::cuda::par.on(stream), thrust::counting_iterator<int>(0), n,
                     _process_masks_op{d_preds, c, n, thr, d_masks, d_score});
}

}  // namespace psenet

}  // namespace mmocr

}  // namespace mmdeploy
