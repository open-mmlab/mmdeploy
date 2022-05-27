// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin
#include <stdint.h>

#include <cub/cub.cuh>

#include "cublas_v2.h"
#include "nms/kernel.h"
#include "trt_plugin_helper.hpp"

#define CUDA_MEM_ALIGN 256

// return cuda arch
size_t get_cuda_arch(int devID) {
  int computeMode = -1, major = 0, minor = 0;
  CUASSERT(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, devID));
  CUASSERT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
  CUASSERT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
  return major * 100 + minor * 10;
}

// ALIGNPTR
int8_t *alignPtr(int8_t *ptr, uintptr_t to) {
  uintptr_t addr = (uintptr_t)ptr;
  if (addr % to) {
    addr += to - addr % to;
  }
  return (int8_t *)addr;
}

// NEXTWORKSPACEPTR
int8_t *nextWorkspacePtr(int8_t *ptr, uintptr_t previousWorkspaceSize) {
  uintptr_t addr = (uintptr_t)ptr;
  addr += previousWorkspaceSize;
  return alignPtr((int8_t *)addr, CUDA_MEM_ALIGN);
}

// CALCULATE TOTAL WORKSPACE SIZE
size_t calculateTotalWorkspaceSize(size_t *workspaces, int count) {
  size_t total = 0;
  for (int i = 0; i < count; i++) {
    total += workspaces[i];
    if (workspaces[i] % CUDA_MEM_ALIGN) {
      total += CUDA_MEM_ALIGN - (workspaces[i] % CUDA_MEM_ALIGN);
    }
  }
  return total;
}

using nvinfer1::DataType;

template <unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta) __global__
    void setUniformOffsets_kernel(const int num_segments, const int offset, int *d_offsets) {
  const int idx = blockIdx.x * nthds_per_cta + threadIdx.x;
  if (idx <= num_segments) d_offsets[idx] = idx * offset;
}

void setUniformOffsets(cudaStream_t stream, const int num_segments, const int offset,
                       int *d_offsets) {
  const int BS = 32;
  const int GS = (num_segments + 1 + BS - 1) / BS;
  setUniformOffsets_kernel<BS><<<GS, BS, 0, stream>>>(num_segments, offset, d_offsets);
}

size_t detectionForwardBBoxDataSize(int N, int C1, DataType DT_BBOX) {
  if (DT_BBOX == DataType::kFLOAT) {
    return N * C1 * sizeof(float);
  }

  printf("Only FP32 type bounding boxes are supported.\n");
  return (size_t)-1;
}

size_t detectionForwardBBoxPermuteSize(bool shareLocation, int N, int C1, DataType DT_BBOX) {
  if (DT_BBOX == DataType::kFLOAT) {
    return shareLocation ? 0 : N * C1 * sizeof(float);
  }
  printf("Only FP32 type bounding boxes are supported.\n");
  return (size_t)-1;
}

size_t detectionForwardPreNMSSize(int N, int C2) {
  ASSERT(sizeof(float) == sizeof(int));
  return N * C2 * sizeof(float);
}

size_t detectionForwardPostNMSSize(int N, int numClasses, int topK) {
  ASSERT(sizeof(float) == sizeof(int));
  return N * numClasses * topK * sizeof(float);
}

size_t detectionInferenceWorkspaceSize(bool shareLocation, int N, int C1, int C2, int numClasses,
                                       int numPredsPerClass, int topK, DataType DT_BBOX,
                                       DataType DT_SCORE) {
  size_t wss[7];
  wss[0] = detectionForwardBBoxDataSize(N, C1, DT_BBOX);
  wss[1] = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DT_BBOX);
  wss[2] = detectionForwardPreNMSSize(N, C2);
  wss[3] = detectionForwardPreNMSSize(N, C2);
  wss[4] = detectionForwardPostNMSSize(N, numClasses, topK);
  wss[5] = detectionForwardPostNMSSize(N, numClasses, topK);
  wss[6] = std::max(sortScoresPerClassWorkspaceSize(N, numClasses, numPredsPerClass, DT_SCORE),
                    sortScoresPerImageWorkspaceSize(N, numClasses * topK, DT_SCORE));
  return calculateTotalWorkspaceSize(wss, 7);
}
