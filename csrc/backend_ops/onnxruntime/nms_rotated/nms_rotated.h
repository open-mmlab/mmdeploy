// Copyright (c) OpenMMLab. All rights reserved.
#ifndef ONNXRUNTIME_NMS_ROTATED_H
#define ONNXRUNTIME_NMS_ROTATED_H

#include <assert.h>
#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <mutex>
#include <string>
#include <vector>

namespace mmdeploy {
struct NMSRotatedKernel {
  NMSRotatedKernel(OrtApi api, const OrtKernelInfo* info);

  void Compute(OrtKernelContext* context);

 private:
  OrtApi api_;
  Ort::CustomOpApi ort_;
  const OrtKernelInfo* info_;
  Ort::AllocatorWithDefaultOptions allocator_;
  float iou_threshold_;
  float score_threshold_;
};

struct NMSRotatedOp : Ort::CustomOpBase<NMSRotatedOp, NMSRotatedKernel> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
    return new NMSRotatedKernel(api, info);
  }
  const char* GetName() const { return "NMSRotated"; }

  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }

  // force cpu
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
};
}  // namespace mmdeploy

#endif  // ONNXRUNTIME_NMS_ROTATED_H
