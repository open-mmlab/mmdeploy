// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin
#include <vector>

#include "nms/kernel.h"

template <typename Dtype, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta) __global__
    void permuteData_kernel(const int nthreads, const int num_classes, const int num_data,
                            const int num_dim, bool confSigmoid, const Dtype *data,
                            Dtype *new_data) {
  // data format: [batch_size, num_data, num_classes, num_dim]
  for (int index = blockIdx.x * nthds_per_cta + threadIdx.x; index < nthreads;
       index += nthds_per_cta * gridDim.x) {
    const int i = index % num_dim;
    const int c = (index / num_dim) % num_classes;
    const int d = (index / num_dim / num_classes) % num_data;
    const int n = index / num_dim / num_classes / num_data;
    const int new_index = ((n * num_classes + c) * num_data + d) * num_dim + i;
    float result = data[index];
    if (confSigmoid) result = exp(result) / (1 + exp(result));

    new_data[new_index] = result;
  }
  // new data format: [batch_size, num_classes, num_data, num_dim]
}

template <typename Dtype>
pluginStatus_t permuteData_gpu(cudaStream_t stream, const int nthreads, const int num_classes,
                               const int num_data, const int num_dim, bool confSigmoid,
                               const void *data, void *new_data) {
  const int BS = 512;
  const int GS = (nthreads + BS - 1) / BS;
  permuteData_kernel<Dtype, BS><<<GS, BS, 0, stream>>>(nthreads, num_classes, num_data, num_dim,
                                                       confSigmoid, (const Dtype *)data,
                                                       (Dtype *)new_data);
  CSC(cudaGetLastError(), STATUS_FAILURE);
  return STATUS_SUCCESS;
}

// permuteData LAUNCH CONFIG
typedef pluginStatus_t (*pdFunc)(cudaStream_t, const int, const int, const int, const int, bool,
                                 const void *, void *);

struct pdLaunchConfig {
  DataType t_data;
  pdFunc function;

  pdLaunchConfig(DataType t_data) : t_data(t_data) {}
  pdLaunchConfig(DataType t_data, pdFunc function) : t_data(t_data), function(function) {}
  bool operator==(const pdLaunchConfig &other) { return t_data == other.t_data; }
};

static std::vector<pdLaunchConfig> pdFuncVec;

bool permuteDataInit() {
  pdFuncVec.push_back(pdLaunchConfig(DataType::kFLOAT, permuteData_gpu<float>));
  return true;
}

static bool initialized = permuteDataInit();

pluginStatus_t permuteData(cudaStream_t stream, const int nthreads, const int num_classes,
                           const int num_data, const int num_dim, const DataType DT_DATA,
                           bool confSigmoid, const void *data, void *new_data) {
  pdLaunchConfig lc = pdLaunchConfig(DT_DATA);
  for (unsigned i = 0; i < pdFuncVec.size(); ++i) {
    if (lc == pdFuncVec[i]) {
      DEBUG_PRINTF("permuteData kernel %d\n", i);
      return pdFuncVec[i].function(stream, nthreads, num_classes, num_data, num_dim, confSigmoid,
                                   data, new_data);
    }
  }
  return STATUS_BAD_PARAM;
}
