// Copyright (c) OpenMMLab. All rights reserved

#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "ort_utils.h"
#include "deform_conv.h"
#include "deform_conv_kernel.hpp"

namespace mmdeploy {

MMCVDeformConvCUDAKernel::MMCVDeformConvCUDAKernel(const OrtApi& api,
                                           const OrtKernelInfo *info)
    : api_(api), ort_(api_), info_(info) {
  std::vector<int64_t> stride =
      ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "stride");
  stride_height_ = stride[0];
  stride_width_ = stride[1];
  std::vector<int64_t> padding =
      ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "padding");
  padding_height_ = padding[0];
  padding_width_ = padding[1];
  std::vector<int64_t> dilation =
      ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "dilation");
  dilation_height_ = dilation[0];
  dilation_width_ = dilation[1];
  deformable_group_ =
      ort_.KernelInfoGetAttribute<int64_t>(info, "deform_groups");
  group_ = ort_.KernelInfoGetAttribute<int64_t>(info, "groups");

  // create allocator
  allocator_ = Ort::AllocatorWithDefaultOptions();

  // create cublas handle
  cublasStatus_t stat = cublasCreate(&m_cublas_handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    ORT_CXX_API_THROW("CUBLAS initialization failed", ORT_FAIL);
  }
}

void MMCVDeformConvCUDAKernel::Compute(OrtKernelContext *context) {
  const int64_t stride_height = stride_height_;
  const int64_t stride_width = stride_width_;
  const int64_t padding_height = padding_height_;
  const int64_t padding_width = padding_width_;
  const int64_t dilation_height = dilation_height_;
  const int64_t dilation_width = dilation_width_;
  const int64_t deformable_group = deformable_group_;
  const int64_t group = group_;

  const OrtValue *input = ort_.KernelContext_GetInput(context, 0);
  const float *input_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<float>(input));

  const OrtValue *offset = ort_.KernelContext_GetInput(context, 1);
  const float *offset_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<float>(offset));

  const OrtValue *filter = ort_.KernelContext_GetInput(context, 2);
  const float *filter_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<float>(filter));

  OrtTensorDimensions input_dims(ort_, input);
  OrtTensorDimensions filter_dims(ort_, filter);

  int64_t batch_size = input_dims[0];
  int64_t in_channels = input_dims[1];
  int64_t in_height = input_dims[2];
  int64_t in_width = input_dims[3];
  int64_t out_channels = filter_dims[0];
  int64_t kernel_height = filter_dims[2];
  int64_t kernel_width = filter_dims[3];

  // get input data type
  auto data_type = ort_.GetTensorElementType(ort_.GetTensorTypeAndShape(input));

  int64_t sizeof_dtype = (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)? sizeof(Ort::Float16_t): sizeof(float);

  // get output memory
  int64_t output_height = floor((in_height + 2 * padding_height -
                              dilation_height * (kernel_height - 1) - 1) /
                                 stride_height +
                             1);
  int64_t output_width = floor(
      (in_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) /
          stride_width +
      1);

  std::vector<int64_t> output_dims = {batch_size, out_channels, output_height,
                                      output_width};

  OrtValue *output = ort_.KernelContext_GetOutput(
      context, 0, output_dims.data(), output_dims.size());
  float *output_ptr = ort_.GetTensorMutableData<float>(output);


  // adopted from trt_deform_conv.cpp -- DeformableConvPluginDynamic::getWorkspaceSize
  int im2col_step = std::min(int64_t(32), batch_size);

  size_t col_size = mmdeploy::getAlignedSize(in_channels * kernel_width * kernel_height * im2col_step * output_height *
                                             output_width * sizeof_dtype);

  size_t out_size = 0;
  if (im2col_step != 1)
    out_size = mmdeploy::getAlignedSize(batch_size * out_channels * output_height * output_width * sizeof_dtype);

  // allocate workspace
  long workspace_size = col_size + out_size;

  float* workspace;
  cudaMalloc(&workspace, workspace_size);
  cudaMemset(workspace, 0, workspace_size);

  auto stream = reinterpret_cast<cudaStream_t>(ort_.KernelContext_GetGPUComputeStream(context));

  switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      deform_conv<float>((float *)input, (float *)filter, (float *)offset, (float *)output_ptr, workspace,
                         batch_size, in_channels, in_height, in_width, out_channels, kernel_width, kernel_height,
                         stride_height, stride_width, padding_height, padding_width, dilation_height,
                         dilation_width, group, deformable_group, im2col_step, m_cublas_handle, stream);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      deform_conv<Ort::Float16_t>((Ort::Float16_t *)input, (Ort::Float16_t *)filter, (Ort::Float16_t *)offset, 
                        (Ort::Float16_t *)output_ptr, workspace, batch_size,
                        in_channels, in_height, in_width, out_channels, kernel_width, kernel_height, stride_height,
                        stride_width, padding_height, padding_width, dilation_height, dilation_width,
                        group, deformable_group, im2col_step, m_cublas_handle, stream);
      break;
    default:
      cudaFree(&workspace);
      ORT_CXX_API_THROW("invalid tensor datatype in deform_conv::CUDAKernel::Compute", ORT_FAIL);
      return;
  }

  cudaFree(&workspace);
}

REGISTER_ONNXRUNTIME_OPS(mmdeploy, MMCVDeformConvCUDAOp);
}  // namespace mmdeploy