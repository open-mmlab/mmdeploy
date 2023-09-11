// Copyright (c) OpenMMLab. All rights reserved.
#ifndef ONNXRUNTIME_NMS_MATCH_H
#define ONNXRUNTIME_NMS_MATCH_H

#include <assert.h>
#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <mutex>
#include <string>
#include <vector>

namespace mmdeploy {
struct NMSMatchKernel {
  NMSMatchKernel(const OrtApi& api, const OrtKernelInfo* info);

  void Compute(OrtKernelContext* context);

 private:
  Ort::CustomOpApi ort_;
  const OrtKernelInfo* info_;
  Ort::AllocatorWithDefaultOptions allocator_;
};

struct NMSMatchOp : Ort::CustomOpBase<NMSMatchOp, NMSMatchKernel> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new NMSMatchKernel(api, info);
  }
  const char* GetName() const { return "NMSMatch"; }

  size_t GetInputTypeCount() const { return 4; }
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

#endif  // ONNXRUNTIME_NMS_MATCH_H
