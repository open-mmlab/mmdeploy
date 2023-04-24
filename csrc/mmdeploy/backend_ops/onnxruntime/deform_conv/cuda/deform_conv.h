// Copyright (c) OpenMMLab. All rights reserved
#ifndef ONNXRUNTIME_DEFORM_CONV_H
#define ONNXRUNTIME_DEFORM_CONV_H

#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

namespace mmdeploy {


struct MMCVDeformConvCUDAKernel {
  MMCVDeformConvCUDAKernel(const OrtApi& api, const OrtKernelInfo *info);

  void Compute(OrtKernelContext *context);

 protected:
  const OrtApi& api_;
  Ort::CustomOpApi ort_;
  const OrtKernelInfo *info_;
  Ort::AllocatorWithDefaultOptions allocator_;

  int cuda_dev_memory_pools_supported_;
  cublasHandle_t cublas_handle_ = nullptr;  // TODO:: release the cublas handle?

  int64_t stride_height_;
  int64_t stride_width_;
  int64_t padding_height_;
  int64_t padding_width_;
  int64_t dilation_height_;
  int64_t dilation_width_;
  int64_t deformable_group_;
  int64_t group_;
  int64_t im2col_step_;

private:
  cudaError_t cuda_malloc(void **pointer, size_t size, cudaStream_t stream);
  void cuda_free(void *pointer, cudaStream_t stream);
};

struct MMCVDeformConvCUDAOp
    : Ort::CustomOpBase<MMCVDeformConvCUDAOp, MMCVDeformConvCUDAKernel> {
  void *CreateKernel(const OrtApi& api, const OrtKernelInfo *info) const {

    return new MMCVDeformConvCUDAKernel(api, info);
  }

  const char *GetName() const { return "DeformableConv2D"; };

  size_t GetInputTypeCount() const { return 3; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(
      size_t index) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }

  size_t GetOutputTypeCount() const { return 1; };

  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  // force CUDA EP
  const char *GetExecutionProviderType() const {
    return "CUDAExecutionProvider";
  };
};

}  // namespace mmdeploy
#endif