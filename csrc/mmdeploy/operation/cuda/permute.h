// Copyright (c) OpenMMLab. All rights reserved.
#ifndef MMDEPLOY_OPERATION_CUDA_PERMUTE_H_
#define MMDEPLOY_OPERATION_CUDA_PERMUTE_H_

#include <cuda_runtime.h>

#include <cstdlib>

namespace mmdeploy {
namespace operation {
namespace cuda {

const int MAX_PERMUTE_DIM = 8;

struct TensorStride {
  int v_[MAX_PERMUTE_DIM];
  int& operator[](size_t idx) { return v_[idx]; }
};

}  // namespace cuda
}  // namespace operation
}  // namespace mmdeploy

#endif  // MMDEPLOY_OPERATION_CUDA_PERMUTE_H_
