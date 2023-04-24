// Copyright (c) OpenMMLab. All rights reserved
#ifndef ONNXRUNTIME_DEFORM_CONV_H
#define ONNXRUNTIME_DEFORM_CONV_H

#include <onnxruntime_cxx_api.h>

namespace mmdeploy {

struct MMCVDeformConvKernel {
  MMCVDeformConvKernel(const OrtApi& api, const OrtKernelInfo *info);
  ~MMCVDeformConvKernel();

  void Compute(OrtKernelContext *context);

 private:
  OrtOp* op_gemm_{};
  void initGemm(Ort::CustomOpApi ort);
  void deformConv(OrtKernelContext *context,
    const float *src, const float *offset, const float *filter,
    const int64_t batch, const int64_t src_c, const int64_t src_h,
    const int64_t src_w, const int64_t dst_c, const int64_t dst_h,
    const int64_t dst_w, const int64_t group, const int64_t offset_group,
    const int64_t channels, const int64_t num_output, const int64_t kernel_h,
    const int64_t kernel_w, const int64_t stride_h, const int64_t stride_w,
    const int64_t pad_h, const int64_t pad_w, const int64_t dilation_h,
    const int64_t dilation_w, float *columns, float *dst);

 protected:
  const OrtApi& api_;
  Ort::CustomOpApi ort_;
  const OrtKernelInfo *info_;
  Ort::AllocatorWithDefaultOptions allocator_;

  int64_t stride_height_;
  int64_t stride_width_;
  int64_t padding_height_;
  int64_t padding_width_;
  int64_t dilation_height_;
  int64_t dilation_width_;
  int64_t deformable_group_;
  int64_t group_;
  int64_t im2col_step_;
};

struct MMCVDeformConvOp
    : Ort::CustomOpBase<MMCVDeformConvOp, MMCVDeformConvKernel> {
  void *CreateKernel(const OrtApi& api, const OrtKernelInfo *info) const {
    return new MMCVDeformConvKernel(api, info);
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

  // force cpu
  const char *GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  };
};

}  // namespace mmdeploy
#endif